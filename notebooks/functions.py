def create_ct_matrix(gene_counts='..\\data\\interim\\df_gene_counts.csv', donor_file='..\\data\\raw\\DonorInformation.csv',tbidata_file='..\\data\\raw\\tbi_data_files.csv', genes='..\\data\\raw\\gene_expression_matrix_2016-03-03\\rows-genes.csv'):
    import pandas as pd
    import numpy as np


    df_gene_counts=pd.read_csv(gene_counts)
    donor=pd.read_csv(donor_file)
    tbidata=pd.read_csv(tbidata_file)
    genes=pd.read_csv(genes)
    merge_table=df_gene_counts.merge(tbidata,left_on='link',right_on='gene_level_fpkm_file_link')

    # Create mapping to map gene id in the count files to gene id used elsewhere
    mapping_dict_gene=dict(zip(genes['gene_entrez_id'],genes['gene_id']))

    #choose fields needed 
    total_counts=merge_table[['expected_count','TPM','FPKM','donor_id', 'donor_name',
        'specimen_id', 'specimen_name','structure_id', 'structure_acronym', 'structure_name',
        'rnaseq_profile_id','gene_id']]
    total_counts['gene_id_mapped']=total_counts['gene_id'].map(mapping_dict_gene)

    #create list to drop genes with no mapping
    unmapped_genes=list(total_counts[total_counts['gene_id_mapped'].isna()]['gene_id'].unique())
    total_counts=total_counts.drop(total_counts[total_counts['gene_id'].isin(unmapped_genes)].index)

    #set fields to integer
    total_counts['expected_count']=np.round(total_counts['expected_count']).astype(int)
    total_counts['gene_id_mapped']=total_counts['gene_id_mapped'].astype(int)

    #pivot table to correct formation
    ct_matrix=total_counts[['gene_id_mapped','rnaseq_profile_id','expected_count']].pivot(index='gene_id_mapped',columns='rnaseq_profile_id', values='expected_count')

    #drop zero reads
    ct_matrix=ct_matrix[(ct_matrix != 0).any(axis=1)] #https://stackoverflow.com/questions/22649693/drop-rows-with-all-zeros-in-pandas-data-frame   
    ct_matrix.to_csv('..\\data\\interim\\ct_matrix.csv')

    exp_design=donor.merge(tbidata,how='left',left_on='donor_id',right_on='donor_id')[['rnaseq_profile_id','act_demented']]
    exp_design['act_demented']=exp_design['act_demented'].str.replace(' ','')
    exp_design=exp_design.set_index('rnaseq_profile_id').loc[list(ct_matrix.columns)] #https://stackoverflow.com/questions/26202926/sorting-a-pandas-dataframe-by-the-order-of-a-list
    exp_design.rename(columns={'act_demented':'condition'}).to_csv('..\\data\\interim\\exp_design.csv')

    return ct_matrix, exp_design

def diff_exp_pipeline(ct_matrix='..\\data\\interim\\ct_matrix.csv', exp_design = '..\\data\\interim\\exp_design.csv', genes='..\\data\\raw\\gene_expression_matrix_2016-03-03\\rows-genes.csv',lv=True,pval_cutoff=.05, lfc_cutoff=0.5, r_installation_folder=None):
    from rnalysis import filtering
    import pandas as pd
    genes=pd.read_csv(genes)

    if lv is True:
        ct_matrix_filter=filtering.CountFilter(ct_matrix,is_normalized=False)

        #normalize to reads per million for limma voom
        ct_matrix_filter.normalize_to_rpm()
        ct_matrix_filter.differential_expression_limma_voom(design_matrix=exp_design,comparisons=[('condition','Dementia','NoDementia')],r_installation_folder=r_installation_folder, output_folder='..\\data\\interim')
        res=pd.read_csv('..\\data\\interim\\LimmaVoom_condition_Dementia_vs_NoDementia.csv')
        res=res.rename(columns={'adj.P.Val':'padj', 'logFC':'lfc', 't':'stat'})

    else:

        #insert deseq workflow
        res=None

    res_filtered=res[((res['padj']<pval_cutoff)&((abs(res['lfc']))>lfc_cutoff))]
    res_genes_merged=res_filtered.merge(genes,left_on='Unnamed: 0',right_on='gene_id')

    return pd.concat([res_genes_merged.sort_values(by='stat').head(10),res_genes_merged.sort_values(by='stat').tail(10)])







