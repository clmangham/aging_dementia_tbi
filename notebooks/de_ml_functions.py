import os
import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


def get_counts():

    # Load raw data
    tbidata=pd.read_csv('../data/raw/tbi_data_files.csv')
    df_gene_counts=pd.read_csv('../data/interim/df_gene_counts.csv')
    genes=pd.read_csv('../data/raw/gene_expression_matrix_2016-03-03/rows-genes.csv')

    # Merge adnd map data
    merge_table=df_gene_counts.merge(tbidata,left_on='link',right_on='gene_level_fpkm_file_link')
    mapping_dict_gene=dict(zip(genes['gene_entrez_id'],genes['gene_id']))

    # Get total counts
    total_counts=merge_table[['expected_count','TPM','FPKM','donor_id', 'donor_name',
        'specimen_id', 'specimen_name','structure_id', 'structure_acronym', 'structure_name',
        'rnaseq_profile_id','gene_id']]
    total_counts['gene_id_mapped']=total_counts['gene_id'].map(mapping_dict_gene)
    unmapped_genes=list(total_counts[total_counts['gene_id_mapped'].isna()]['gene_id'].unique())
    total_counts=total_counts.drop(total_counts[total_counts['gene_id'].isin(unmapped_genes)].index)
    total_counts['expected_count']=np.round(total_counts['expected_count']).astype(int)
    total_counts['gene_id_mapped']=total_counts['gene_id_mapped'].astype(int)

    # Generate counts table
    ct_matrix=total_counts[['gene_id_mapped','rnaseq_profile_id','expected_count']].pivot(index='gene_id_mapped',columns='rnaseq_profile_id', values='expected_count')

    return ct_matrix

def prep_data():
    """
    Prep dataframe for ML models
    """
    samples = pd.read_csv('../data/raw/gene_expression_matrix_2016-03-03/columns-samples.csv')
    donor_info = pd.read_csv('../data/raw/DonorInformation.csv')

    # Process donor info to segregate control group
    control_group_df = donor_info[donor_info['act_demented'] == 'No Dementia']
    dementia_group_df = donor_info[donor_info['act_demented'] != 'No Dementia']

    # Get donor ids
    control_ids = control_group_df['donor_id']
    dementia_ids = dementia_group_df['donor_id']

    # Assign condition to sample data
    samples['Condition'] = samples['donor_id'].apply(lambda x: 'control' if x in control_ids.values else 'dementia')

    # Grab donor ids
    donor_ids = samples.donor_id.unique()

    # Create ML dataframe
    counts = pd.read_pickle('../data/interim/PyDeseq2/ct_matrix.pkl')
    top_genes = pd.read_pickle('../data/interim/PyDeseq2/top_bottom_ten_sigs.pkl')
    lv_genes = pd.read_pickle('../data/interim/lv_genes')

    # Create gene list
    gene_list = pd.concat((top_genes.gene_id, lv_genes.iloc[:,0])).drop_duplicates().values

    # Filter by top genes (top and bottom 10)
    ml_df = counts[counts.index.isin(gene_list)].copy()

    # Transform and apply conditions (y_label)
    ml_df = ml_df.T.reset_index().rename_axis(None, axis = 1)
    ml_df['Condition'] = samples['Condition']

    return ml_df, donor_ids, samples

def train_test_val_split(ml_df ,donor_ids, samples):
    # 70, 15, 15 Train, Validate, Test split
    # np.random.seed(42)
    # train_ids = np.random.choice(donor_ids, int(np.ceil(len(donor_ids) * 0.7)))
    # test_ids = np.setdiff1d(donor_ids, train_ids)
    # validate_ids = np.random.choice(test_ids, int(np.ceil(len(test_ids) * 0.5)))
    # test_ids = np.setdiff1d(test_ids, validate_ids)

    train_ids, test_ids = train_test_split(donor_ids, test_size=0.33, random_state= 42)
    train_ids, validate_ids = train_test_split(train_ids, test_size=0.33, random_state= 42)

    #samples (rna_profile_ids) by donor data splt
    train_samples = samples[samples['donor_id'].isin(train_ids)]['rnaseq_profile_id']
    val_samples = samples[samples['donor_id'].isin(validate_ids)]['rnaseq_profile_id']
    test_samples = samples[samples['donor_id'].isin(test_ids)]['rnaseq_profile_id']

    # Now can filter by train, val, test, split
    train_df = ml_df[ml_df['rnaseq_profile_id'].isin(train_samples)].drop(columns = 'rnaseq_profile_id')
    val_df = ml_df[ml_df['rnaseq_profile_id'].isin(val_samples)].drop(columns = 'rnaseq_profile_id')
    test_df = ml_df[ml_df['rnaseq_profile_id'].isin(test_samples)].drop(columns = 'rnaseq_profile_id')

    # final data prep
    X_train = train_df.drop(columns='Condition')
    y_train = train_df['Condition']

    X_val = val_df.drop(columns='Condition')
    y_val = val_df['Condition']

    X_test = test_df.drop(columns='Condition')
    y_test = test_df['Condition']

    # Scale data and transform data
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    y_train = y_train.apply(lambda x: 1 if x == 'dementia' else 0)
    y_val = y_val.apply(lambda x: 1 if  x=='dementia' else 0)
    y_test = y_test.apply(lambda x: 1 if  x=='dementia' else 0)

    return X_train, y_train, X_val, y_val, X_test, y_test

def custom_train_test_split(ml_df ,donor_ids, samples):
    # 70, 30 Train, Test split
    # np.random.seed(42)
    train_ids, test_ids = train_test_split(donor_ids, test_size=0.33)
    # print(len(train_ids), len(test_ids))

    #samples (rna_profile_ids) by donor data splt
    train_samples = samples[samples['donor_id'].isin(train_ids)]['rnaseq_profile_id']
    test_samples = samples[samples['donor_id'].isin(test_ids)]['rnaseq_profile_id']

    # Now can filter by train, val, test, split
    train_df = ml_df[ml_df['rnaseq_profile_id'].isin(train_samples)].drop(columns = 'rnaseq_profile_id')
    test_df = ml_df[ml_df['rnaseq_profile_id'].isin(test_samples)].drop(columns = 'rnaseq_profile_id')

    # final data prep
    X_train = train_df.drop(columns='Condition')
    y_train = train_df['Condition']

    X_test = test_df.drop(columns='Condition')
    y_test = test_df['Condition']

    # Scale data and transform data
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = y_train.apply(lambda x: 1 if x == 'dementia' else 0)
    y_test = y_test.apply(lambda x: 1 if  x=='dementia' else 0)

    return X_train, y_train, X_test, y_test


def train_models(models, model_names, X_train, y_train, X_val, y_val):
    score_names = ['accuracy',
                   'precision',
                   'recall',
                   'f1',
                    #   'roc_auc'
                      ]
    scores = []
    for name, model in zip(model_names, models):
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        accuracy_scores = accuracy_score(y_val, preds)
        precision_scores = precision_score(y_val, preds)
        recall_scores = recall_score(y_val, preds)
        f1_scores = f1_score(y_val,preds)
        # roc_auc_scores = roc_auc_score(y_val,preds)

        scores.append([accuracy_scores,
                       precision_scores,
                       recall_scores,
                       f1_scores,
                        # roc_auc_scores
                        ])

    model_scores = pd.DataFrame(scores, index= model_names, columns= score_names)

    return model_scores


def train_models_boot_data(ml_df,donor_ids, samples, models, model_names, iterations=1000):
    score_names = ['Accuracy',
                   'Precision',
                   'Recall',
                   'F1',
                      ]
    scores_accuracy = []
    scores_precision = []
    scores_recall = []
    scores_f1 = []
    for i in range(iterations + 1):
        X_train, y_train, X_test, y_test = custom_train_test_split(ml_df,donor_ids, samples)
        scores_accuracy_i = []
        scores_precision_i = []
        scores_recall_i = []
        scores_f1_i = []
        for name, model in zip(model_names, models):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            scores_accuracy_i.append(accuracy_score(y_test, preds))
            scores_precision_i.append(precision_score(y_test, preds))
            scores_recall_i.append(recall_score(y_test, preds))
            scores_f1_i.append(f1_score(y_test,preds))

        scores_accuracy.append(scores_accuracy_i)
        scores_precision.append(scores_precision_i)
        scores_recall.append(scores_recall_i)
        scores_f1.append(scores_f1_i)

    scores_list = [scores_accuracy, scores_precision, scores_recall, scores_f1]
    model_scores_dict = {}
    for name, scores in zip(score_names, scores_list):
        model_scores_dict[name] = pd.DataFrame(scores, columns= model_names)

    return model_scores_dict

def plot_model_scores(model_scores, ml_df, score_name):
    sort_order = np.argsort(model_scores.mean().values)
    descending_sort = sort_order[::-1]
    sorted_labels = model_scores.iloc[:,descending_sort].columns

    plt.figure(figsize=(10, 6))
    sns.boxplot(model_scores.iloc[:,descending_sort], orient= 'h',palette='Spectral')
    y_labels = sorted_labels
    plt.yticks(range(len(y_labels)), y_labels)
    # plt.title('Models Trained on Bootstrapped Data')
    plt.xlabel(f'{score_name} Score')
    plt.ylabel('Gene Symbol')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()

def subplot_plot_model_scores(model_scores, ml_df, score_name, ax=None):
    sort_order = np.argsort(model_scores.mean().values)
    descending_sort = sort_order[::-1]
    sorted_labels = model_scores.iloc[:, descending_sort].columns

    # Use the provided subplot or create a new one
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

    sns.boxplot(model_scores.iloc[:, descending_sort], orient='h', palette='Spectral', ax=ax)
    y_labels = sorted_labels
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel(f'{score_name} Score')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvline(0.5,color="gray",linestyle="--")