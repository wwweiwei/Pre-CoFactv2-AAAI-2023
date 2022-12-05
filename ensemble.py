import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.metrics import classification_report

model_path = {
    'wei':'./ensemble_model/Validation/wei_val_answer_prob.csv',
    'hwu':'./ensemble_model/Validation/hwu_val_answer_prob.csv',
    'yao':'./ensemble_model/Validation/yao_val_answer_prob.csv'
}
model_count = len(model_path.keys())

inverse_category = {
    0: 'Support_Multimodal',
    1: 'Support_Text',
    2: 'Insufficient_Multimodal',
    3: 'Insufficient_Text',
    4: 'Refute'
}


def load_data(mode = 'val'):
    df = {}
    for key, value in model_path.items():
        df[key] = pd.read_csv(model_path[key])
    
    if mode == 'val':
        gt = pd.read_csv('./data/val.csv', index_col=0, sep='\t')[['Category']]
    else:
        gt = pd.read_csv('./data/test.csv', index_col=0, sep='\t')[['Category']]
    
    return df, gt


def grid_search_weight_and_power(df, gt):
    best_f1 = 0
    weight = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    power = [1/8, 1/4, 1/2, 1, 2]
    best_w1, best_w2, best_w3, best_p1, best_p2, best_p3 = 0, 0, 0, 0, 0, 0

    for w1 in tqdm(weight):
        for w2 in weight:
            for w3 in weight:
                for p1 in power:
                    for p2 in power:
                        for p3 in power:
                            answer = []
                            for i in range(len(df['wei'])):
                                prob = []
                                if model_count == 1:
                                    for prob_1 in zip(df['wei'].iloc[i].values.tolist()[1:]):
                                        prob.append(prob_1)
                                else:
                                    # 0 is index in dataframe
                                    for prob_1, prob_2, prob_3 in \
                                        zip(df['wei'].iloc[i].values.tolist()[1:], df['hwu'].iloc[i].values.tolist()[1:], df['yao'].iloc[i].values.tolist()[1:]):
                                        current_prob = (prob_1**p1) * w1 + (prob_2**p2) * w2 + (prob_3**p3) * w3
                                        prob.append(current_prob)

                                category = prob.index(max(prob))
                                answer.append([i+1, inverse_category[category]])

                            ans_df = pd.DataFrame(answer, columns =['id', 'Category'])
                            ans_df = ans_df.set_index('id')

                            f1_score_ = round(f1_score(gt, ans_df[['Category']], average='weighted'), 5)
                            if f1_score_ > best_f1:
                                best_f1 = f1_score_
                                best_w1, best_w2, best_w3, best_p1, best_p2, best_p3 = w1, w2, w3, p1, p2, p3
                                print('----------')
                                print(f'best_f1: {best_f1}')
                                print(f'w1: {w1}, w2: {w2}, w3: {w3}, p1: {p1}, p2: {p2}, p3: {p3}')

    return best_w1, best_w2, best_w3, best_p1, best_p2, best_p3


def get_ensemble_pred(w1, w2, w3, p1, p2, p3, df, gt):
    answer = []
    for i in range(len(df['wei'])):
        prob = []
        if model_count == 1:
            for prob_1 in zip(df['wei'].iloc[i].values.tolist()[1:]):
                prob.append(prob_1)
        else:
            # 0 is index in dataframe
            for prob_1, prob_2, prob_3 in \
                zip(df['wei'].iloc[i].values.tolist()[1:], df['hwu'].iloc[i].values.tolist()[1:], df['yao'].iloc[i].values.tolist()[1:]):
                current_prob = (prob_1**p1) * w1 + (prob_2**p2) * w2 + (prob_3**p3) * w3
                prob.append(current_prob)

        category = prob.index(max(prob))
        answer.append([i+1, inverse_category[category]])

    pred = pd.DataFrame(answer, columns =['id', 'Category']).set_index('id')
    print(classification_report(pred, gt, target_names=None, digits=4))

    SAVE = False
    if SAVE == True:
        assert len(answer) == len(df['wei'])
        pred.to_csv('pred.csv', index=False)

    return pred


def draw_confusion_matrix(pred, gt):
    font = {'size'   : 5}
    plt.rc('font', **font)
    labels = ['Support_Multimodal', 'Support_Text', 'Insufficient_Multimodal', 'Insufficient_Text', 'Refute']
    a = confusion_matrix(gt, pred['Category'], labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=a, display_labels=labels).plot(cmap='cividis')
    disp.figure_.savefig('val_confusion_matrix.png', dpi=1000)


df, gt = load_data('val')
# best_w1, best_w2, best_w3, best_p1, best_p2, best_p3 = grid_search_weight_and_power(df, gt)
best_w1, best_w2, best_w3, best_p1, best_p2, best_p3 = 0.2, 0.7, 0.6, 0.125, 0.125, 0.25

pred = get_ensemble_pred(best_w1, best_w2, best_w3, best_p1, best_p2, best_p3, df, gt)
draw_confusion_matrix(pred, gt)