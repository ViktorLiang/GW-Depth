import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Set GlassRGBD arguments', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--save_freq', default=1, type=int)
    parser.add_argument('--input_log_freq', default=0.98, type=int)
    
    parser.add_argument('--benchmark', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--append_word', default=None, type=str, help="Name of the convolutional backbone to use")
    # Model parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # Load
    parser.add_argument('--layer1_frozen', action='store_true')
    parser.add_argument('--layer2_frozen', action='store_true')

    parser.add_argument('--frozen_weights', default='', help='resume from checkpoint')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--no_opt', action='store_true')

    # Transformer
    parser.add_argument('--LETRpost', action='store_true')
    parser.add_argument('--layer1_num', default=3, type=int)
    parser.add_argument('--layer2_num', default=2, type=int)

    # line detection Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=1000, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_line', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_point', default=5, type=float,
                        help="L1 box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--point_loss_coef', default=5, type=float)
    parser.add_argument('--line_loss_coef', default=5, type=float)
    parser.add_argument('--line_depth_loss_coef', default=0.3, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float) # line label loss coef for background
    parser.add_argument('--label_loss_func', default='cross_entropy', type=str)
    parser.add_argument('--label_loss_params', default='{}', type=str)
    parser.add_argument('--variance_focus', type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)
    parser.add_argument('--log_depth_error', action='store_true')
    parser.add_argument('--with_plane_norm_loss', action='store_true')
    parser.add_argument('--plane_norm_loss_coef', default=50, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
   

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--num_gpus', default=1, type=int)


    parser.add_argument('--data_path', type=str, default='/home/ly/data/datasets/trans-depth/Glass-RGBD-Dense/images/', help='image path', required=False)
    parser.add_argument('--gt_depth_path', type=str, default='/home/ly/data/datasets/trans-depth/Glass-RGBD-Dense/depth/', help='ground truth depth path', required=False)
    parser.add_argument('--gt_seg_path', type=str, default='/home/ly/data/datasets/trans-depth/Glass-RGBD-Dense/segmentation/', help='ground truth segmentation path', required=False)
    parser.add_argument('--gt_line_path', type=str, default='/home/ly/data/datasets/trans-depth/Glass-RGBD-Dense/polygon_json/', help='ground truth of line path', required=False)
    parser.add_argument('--filenames_file_train', type=str, default='/home/ly/data/datasets/trans-depth/Glass-RGBD-Dense/train_182.txt', help='train names file', required=False)
    parser.add_argument('--filenames_file_eval', type=str, default='/home/ly/data/datasets/trans-depth/Glass-RGBD-Dense/val_182.txt', help='evaluation names file', required=False)
    parser.add_argument('--glassrgbd_images_json', type=str, default='/home/ly/data/datasets/trans-depth/Glass-RGBD-Dense/glassrgbd_images.json', help='image information json', required=False)

    parser.add_argument('--glassrgbd_rhint_path', type=str, default='/mnt/lab/data/depth/glass-rgbd-rhint/', help='glass reflection points in json file', required=False)
    parser.add_argument('--glassrgbd_rhint_points_path', type=str, default='/mnt/lab/data/depth/glass-rgbd-rpoints-1_5/', help='glass reflection points in json file', required=False)
    parser.add_argument('--rhint_max_ratio', default=[1/3, 1/5, 1/12], type=list, help='sample ratio corresponding to max reflection values')
    parser.add_argument('--num_reflection_points', default=50, type=int, help='reference points num')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--gpu_ids', default=[0], type=list)

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--dataset', default='train', type=str, choices=('train', 'val'))
    parser.add_argument('--dataset_args_file', default='', type=str)

    # depth, segmentation
    parser.add_argument('--with_line', action='store_true')
    parser.add_argument('--with_dense', action='store_true')
    parser.add_argument('--with_center', action='store_true')
    parser.add_argument('--with_reflection', action='store_true')
    parser.add_argument('--with_dense_center', action='store_true', help='whether to use center points in dense decoders')
    parser.add_argument('--with_line_depth', action='store_true', help='whether to predict line points depth') 
    
    parser.add_argument('--max_depth', default=10, type=int)
    parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=10)
    parser.add_argument('--dense_trans_dim', default=512, type=int, help='dense transformer dim')
    parser.add_argument('--dense_trans_layers', default=[4,], type=list, help='dense transformer layers with depth')
    parser.add_argument('--dense_trans_heads', default=16, type=list, help='heads number of dense transformer')
    parser.add_argument('--class_trans_layers', default=[2, 2, 1], type=list, help='class transformer layers with depth')
    parser.add_argument('--group_attention_layers', default=[[False, False], [False, False], [False]], type=list, help='class attention layers with point attention or not')
    parser.add_argument('--class_init_size', default=32, type=int, help='dense transformer dim')
    parser.add_argument('--depth_interval', default=[0.1, 0.3, 0.5, 0.7, 0.9], type=list, help='depth interval')
    #parser.add_argument('--depth_interval', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], type=list, help='depth interval')
    #parser.add_argument('--depth_interval', default=[0.1, 0.3, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9], type=list, help='depth interval')
    #parser.add_argument('--depth_interval', default=[0.1, 0.5, 0.9], type=list, help='depth interval')
    #parser.add_argument('--depth_interval', default=[0.5], type=list, help='depth interval')
    #parser.add_argument('--depth_interval', default=[], type=list, help='depth interval')
    parser.add_argument('--depth_sample_layers', default=[True, True, True], type=list, help='class attention layers with prediction or not')
    parser.add_argument('--interval_sample_num', default=[30, 80, 160], type=list, help='sample centain number')
    #parser.add_argument('--interval_sample_num', default=[40, 100, 0], type=list, help='sample centain number')

    parser.add_argument('--top_trans_points', action='store_true', help='whether to use points guided reweight layer at top transformer layer')
    parser.add_argument('--with_point_attention', action='store_true', help='whether to use points attention')
    parser.add_argument('--class_tokenfuse_layers', default=[False, False, False], type=list, help='class attention layers with prediction or not')
    #parser.add_argument('--depth_pred_layers', default=[True, True, True], type=list, help='class attention layers with prediction or not')
    parser.add_argument('--depth_pred_layers', default=[False, False, False], type=list, help='class attention layers with prediction or not')
    parser.add_argument('--points_double_layers', default=[False, False, False], type=list, help='whether to double the sample points')
    parser.add_argument('--points_inline_sample_layers', default=[False, False, False], type=list, help='whether to double the sample points')

    parser.add_argument('--class_token_dim', default=64, type=int, help='class token dim')
    #parser.add_argument('--num_ref', default=28, type=int, help='reference lines num')
    parser.add_argument('--num_ref', default=20, type=int, help='reference lines num')
    parser.add_argument('--pooling_base_size', default=1024, type=int, help='pooling base size')
    parser.add_argument('--pooling_max_size', default=64, type=int, help='adaptive pooling size')
    parser.add_argument('--adaptive_min_ratio', default=1/16, type=float, help='pooling min ratio')
    parser.add_argument('--fix_line_model', action='store_true')
    parser.add_argument('--depth_loss_weights', default=[1/4, 1/4, 1/4, 1], type=list, help='loss weight for depth of different layers')
    parser.add_argument('--seg_loss_weight', default=2.0, type=float, help='loss weight for segmentation')
    # parser.add_argument('--num_clusters', default=16, type=int, help='number of centers for one image, represent the instance number of glasses')
    # parser.add_argument('--shortest_ratio', default=0.08, type=int, help='shortest ratio of image size for sampled lines')


    return parser
