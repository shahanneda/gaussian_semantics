#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage


output_path = ""
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    do_instance_only = False
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    # gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset)

    gaussians = scene.gaussians
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    def export_instance_image(image, i):
        # print(torch.unique(image))
        image = image / torch.max(image)
        # print(torch.unique(image))
        # print(torch.max(image))
        to_pil = ToPILImage()
        image_bw = to_pil(image)
        image_bw.save(f"test_output/instance_image-{i}.jpg")

    def do_instance_iterations(iterations=100):
        for iteration in range(iterations):
            viewpoint_stack = None
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            bg = torch.rand((3), device="cuda") if opt.random_background else background
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, should_do_instance=True)
            instance_images = render_pkg["instance_images"]

            # print(torch.unique(gaussians.get_instance))
            # print(gaussians.get_instance)
            gt_instance_image = viewpoint_cam.instance_image.cuda()
            export_instance_image(gt_instance_image, 5)
            gt_labels = torch.flatten(gt_instance_image, 1, 2)
        # print(torch.unique(gt_labels))
            # gt_labels = gt_labels * 4
            gt_labels = gt_labels.to(torch.int64)
            gt_labels = gt_labels.reshape((-1))

            # print(instance_images[2])

            if iteration % 2 == 0:
                for i, instance_image in enumerate(instance_images):
                    export_instance_image(instance_image, i)

            instance_tensor = torch.cat(instance_images)
            instance_tensor = instance_tensor.flatten(1, 2)
            instance_tensor = instance_tensor.T
            instance_cel = torch.nn.CrossEntropyLoss()
            instance_loss = instance_cel(instance_tensor, gt_labels)
            instance_loss.backward()
            print(f"Doing Instance Iteration: {iteration}", instance_loss)
            with torch.no_grad():
                    gaussians.instance_optimizer.step()
                    gaussians.instance_optimizer.zero_grad(set_to_none = True)

    # do_instance_iterations()
    # scene.save(1000)
    # print(f"\nTraining complete. Saved to {output_path}")
    # if do_instance_only:
    #     print("Doing instance optimization only!")
    #     for iteration in range(first_iter, opt.iterations + 1):        
    #         if (iteration in saving_iterations):
    #             print("\n[ITER {}] Saving Gaussians".format(iteration))
    #             scene.save(iteration)

    if do_instance_only:
        do_instance_iterations(1000)
        print(f"\nTraining complete. Saved to {output_path}")
        scene.save(1000)
        return

    # torch.autograd.set_detect_anomaly(True)
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, should_do_instance=False)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # print("image shape is", image.permute(2, 1, 0).shape)

        # print(f"mean instnace {torch.mean(gaussians.get_instance)} max instance {torch.max(gaussians.get_instance)} mode ins {torch.mode(gaussians.get_instance)}")
        # print(f"ins min {torch.min(instance_image)}")
        # print(f"ins max {torch.max(instance_image)}")
        # print(f"ins unique {torch.unique(instance_image)}")
        # print("gt: ", torch.unique(gt_instance_image))
        # print("it: ", torch.unique(instance_image))
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        # Ll1_instance = l1_loss(instance_image, gt_instance_image)
        # Ll1_instance = 0 
        lambda_instance_loss = 0.8




        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) #+ lambda_instance_loss*instance_loss

        # print(f"Instance loss is {Ll1_instance}")
        loss.backward()


        # print("instance image shape: ", instance_image.shape)
        # plt.imsave("test_output/image.jpg", image.permute(1, 2, 0).cpu().detach().numpy());

        if iteration % 200 == 0:
            plt.imsave(f"test_output/image.jpg", image.permute(1, 2, 0).clamp(0, 1).cpu().detach().numpy());
            # plt.imsave(f"test_output/image.jpg", image.permute(1, 2, 0).clamp(0, 1).cpu().detach().numpy());
            # export_instance_image(instance_image.cpu().detach())
        # export_instance_image(gt_instance_image)

        if iteration > 200 and iteration % 200 == 0:
            do_instance_iterations()



        # plt.imsave("test_output/image.jpg", gt_instance_image.permute(2, 1, 0).cpu().detach().numpy(), cmap="gray");
        # plt.imsave("test_output/instance_image.jpg", image, cmap="gray");

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

        with torch.no_grad():
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

        # TODO: remove this
        global output_path
        output_path = args.model_path
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print(f"\nTraining complete. Saved to {output_path}")
