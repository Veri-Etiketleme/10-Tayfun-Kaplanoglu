from action import test_with_dnn, test_svm, test_xgboost
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='basketball game prediction')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 1)')
    parser.add_argument('--epoch', type=int, default=10,
                    help='epoch number')               
    parser.add_argument('--cuda', type=int, default=1,
                    help='CUDA training')
    parser.add_argument('--model', type=str, default="dnn",
                    help='Choose model (dnn=0,svm=1,xgboost=2)')                           
    parser.add_argument('--model_name', type=str, default='epoch_30_params.pkl',
                    help='model name')
    parser.add_argument('--team_data_type', type=str, default='average',
                    help='team data type')
    args = parser.parse_args()
    if args.model == "dnn":
        test_with_dnn(args)
    elif args.model == "svm":
        test_svm(args)
    elif args.model == "xgboost":
        test_xgboost(args)