import os
import argparse

def parse_scores(filepath):
    results = {}
    with open(filepath, 'r') as scores_file:
        lines = [line.replace(os.linesep, '') for line in scores_file.readlines()]
        break_ind = None
        for i, line in enumerate(lines):
            if line == '==================================================':
                break_ind = i
        lines = lines[break_ind + 2:]
        for line in lines:
            tokens = line.replace(' ', '\t').split('\t')
            subset = f'{tokens[0]}_rmse'
            lambda_coef = float(tokens[1])
            score = float(tokens[3].replace(':', "=").split("=")[1])
            if lambda_coef not in results:
                results[lambda_coef] = {}
            results[lambda_coef][subset] = score
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores_path', type=str, required=True)
    parser.add_argument('--get_best_lambda', action='store_true', required=False)
    parser.add_argument('--get_lambda_score', type=str, required=False)
    args = parser.parse_args()
    scores = parse_scores(args.scores_path)
    best_lambda = None    
    best_score = None
    if args.get_best_lambda:
        for lambda_coef, scores in scores.items():
            if best_score is None or best_score > scores['test_rmse']:
                best_score = scores['test_rmse']
                best_lambda = lambda_coef
        print(best_lambda)
    if args.get_lambda_score:
        target_lambda = float(args.get_lambda_score)
        closest_lambda = None
        for lambda_coef in scores.keys():
            if closest_lambda is None or abs(closest_lambda - target_lambda) > abs(lambda_coef - target_lambda):
                closest_lambda = lambda_coef
        print(scores[closest_lambda]['test_rmse'])
        
