import train_multi_task as mt_train


def get_arg_parser():
    parser = mt_train.get_arg_parser()
    parser.add_argument('-lf', '--load_folder', default='pretrained/multi_task/mt_factor_5to10_ttm/', type=str,
                        help='path to a pretrained model')
    parser.add_argument('-nt', '--new_tasks', nargs='+', type=str, dest='tasks',
                        help='which new tasks to learn!')
    parser.add_argument('-tl','--train_layers', nargs='+',type=str,
                        help="List of layers for retraining, the remaining layers will be frozen!")
    return parser



if __name__ == '__main__':
    args = mt_train.handle_command_line(get_arg_parser())
    mt_train.torch.set_num_threads(1)
    mt_train.main(args)

