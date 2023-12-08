import pandas as pd
from rnalysis import filtering
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import ward, dendrogram
import mpl_axes_aligner
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, HDBSCAN
import sklearn.metrics
import kmedoids
from sklearn.decomposition import PCA
import plotly.express as px
from matplotlib.pyplot import gcf
import gseapy as gp
from gseapy import barplot, dotplot
import matplotlib.ticker as mtick


# this function loads files and transforms them to output a count matrix and tpm matrix"
def create_ct_matrix_tpm_matrix(
    gene_counts="..\\data\\interim\\df_gene_counts.csv",
    donor_file="..\\data\\raw\\DonorInformation.csv",
    tbidata_file="..\\data\\raw\\tbi_data_files.csv",
    genes="..\\data\\raw\\gene_expression_matrix_2016-03-03\\rows-genes.csv",
):

    # read in necessary files
    df_gene_counts = pd.read_csv(gene_counts)
    donor = pd.read_csv(donor_file)
    tbidata = pd.read_csv(tbidata_file)
    genes = pd.read_csv(genes)
    merge_table = df_gene_counts.merge(
        tbidata, left_on="link", right_on="gene_level_fpkm_file_link"
    )

    # Create mapping to map gene id in the count files to gene id used elsewhere
    mapping_dict_gene = dict(zip(genes["gene_entrez_id"], genes["gene_id"]))

    # choose fields needed
    total_counts = merge_table[
        [
            "expected_count",
            "TPM",
            "FPKM",
            "donor_id",
            "donor_name",
            "specimen_id",
            "specimen_name",
            "structure_id",
            "structure_acronym",
            "structure_name",
            "rnaseq_profile_id",
            "gene_id",
        ]
    ]
    total_counts["gene_id_mapped"] = total_counts["gene_id"].map(mapping_dict_gene)

    # create list to drop genes with no mapping
    unmapped_genes = list(
        total_counts[total_counts["gene_id_mapped"].isna()]["gene_id"].unique()
    )
    total_counts = total_counts.drop(
        total_counts[total_counts["gene_id"].isin(unmapped_genes)].index
    )

    # set fields to integer
    total_counts["expected_count"] = np.round(total_counts["expected_count"]).astype(
        int
    )
    total_counts["gene_id_mapped"] = total_counts["gene_id_mapped"].astype(int)

    # pivot table to correct formation
    ct_matrix = total_counts[
        ["gene_id_mapped", "rnaseq_profile_id", "expected_count"]
    ].pivot(
        index="gene_id_mapped", columns="rnaseq_profile_id", values="expected_count"
    )
    tpm_matrix = total_counts[["gene_id_mapped", "rnaseq_profile_id", "TPM"]].pivot(
        index="gene_id_mapped", columns="rnaseq_profile_id", values="TPM"
    )

    # drop zero reads
    ct_matrix = ct_matrix[
        (ct_matrix != 0).any(axis=1)
    ]  # https://stackoverflow.com/questions/22649693/drop-rows-with-all-zeros-in-pandas-data-frame
    tpm_matrix = tpm_matrix[(tpm_matrix != 0).any(axis=1)]
    ct_matrix.to_csv("..\\data\\interim\\ct_matrix.csv")
    tpm_matrix.to_csv("..\\data\\interim\\tpm_matrix.csv")

    # create experiment design data frame
    exp_design = donor.merge(
        tbidata, how="left", left_on="donor_id", right_on="donor_id"
    )[["rnaseq_profile_id", "act_demented"]]
    exp_design["act_demented"] = exp_design["act_demented"].str.replace(" ", "")
    exp_design = exp_design.set_index("rnaseq_profile_id").loc[
        list(ct_matrix.columns)
    ]  # https://stackoverflow.com/questions/26202926/sorting-a-pandas-dataframe-by-the-order-of-a-list
    exp_design.rename(columns={"act_demented": "condition"}).to_csv(
        "..\\data\\interim\\exp_design.csv"
    )

    # return all df
    return ct_matrix, exp_design, tpm_matrix, df_gene_counts, genes, tbidata, donor


# this function makes an api call to get other data from the RNA sequencing analysis
def create_gene_counts_df(tbifile="..\\data\\raw\\tbi_data_files.csv"):
    tbidata = pd.read_csv(tbifile)
    file_links = tbidata["gene_level_fpkm_file_link"].to_list()
    for link in file_links:
        if link == file_links[0]:
            df_gene_counts = pd.read_csv(
                "http://api.brain-map.org" + link, delimiter="\\t", engine="python"
            )
            df_gene_counts["link"] = link
        else:
            file_link = "http://api.brain-map.org" + link
            new_file = pd.read_csv(file_link, delimiter="\\t", engine="python")
            new_file["link"] = link
            df_gene_counts = pd.concat([df_gene_counts, new_file])
    df_gene_counts.to_csv("..\\data\\interim\\df_gene_counts.csv")
    return df_gene_counts


# this function performs limma voom differential expression
def lv_diff_exp(
    pval_cutoff,
    lfc_cutoff,
    genes,
    ct_matrix="..\\data\\interim\\ct_matrix.csv",
    design_matrix="..\\data\\interim\\exp_design.csv",
    r_path="C:\Program Files\R\R-4.3.2",
    output_folder="..\\data\\interim",
    res="..\\data\\interim\\LimmaVoom_condition_Dementia_vs_NoDementia.csv",
):
    ct_matrix_filter = filtering.CountFilter(
        ct_matrix,
        is_normalized=False,
    )
    # normalize to reads per million for limma voom
    ct_matrix_filter.normalize_to_rpm()

    # perform limma voom
    ct_matrix_filter.differential_expression_limma_voom(
        design_matrix=design_matrix,
        comparisons=[("condition", "Dementia", "NoDementia")],
        r_installation_folder=r_path,
        output_folder=output_folder,
    )

    # read results back in
    lv_res = pd.read_csv(res)

    # filter results to signficant values
    lv_filtered = lv_res[
        ((lv_res["adj.P.Val"] < pval_cutoff) & ((abs(lv_res["logFC"])) > lfc_cutoff))
    ]
    print("Significant limma voom results: {}".format(len(lv_filtered)))

    # create and save all DEGs and top and bottom 10 DEGs
    lv_genes_merged = lv_filtered.merge(genes, left_on="Unnamed: 0", right_on="gene_id")
    lv_genes_merged.to_pickle("..//data//interim//all_lv_degs.pkl")
    lv_genes_merged = lv_genes_merged.rename(
        columns={"adj.P.Val": "padj", "logFC": "lfc", "t": "stat"}
    )
    lv_genes = pd.concat(
        [
            lv_genes_merged.sort_values(by="stat").head(10),
            lv_genes_merged.sort_values(by="stat").tail(10),
        ]
    )
    lv_genes.to_pickle("..//data//interim//lv_genes.pkl")

    return lv_genes_merged, lv_genes, lv_res


# this creates mapping dictionaries to use for clustering
def create_mapping_dicts(
    lv,
    genes,
    deseq2="..//data//interim//PyDeseq2//top_bottom_ten_shrunk_sigs.pkl",
):

    deseq2 = pd.read_pickle(deseq2)

    # get all gene ids to filter
    genes["gene_id"] = genes["gene_id"].astype(str)
    deseq2 = deseq2.merge(genes, on="gene_id", how="inner")
    gene_ids = set(
        list([int(x) for x in deseq2["gene_id"]]) + [x for x in list(lv["gene_id"])]
    )
    # get symbols for gene ids
    symbol_map = {
        **dict(zip(lv["gene_id"].astype(int), lv["gene_symbol"])),
        **dict(zip(deseq2["gene_id"].astype(int), deseq2["gene_symbol"])),
    }  # https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-in-python

    # get gene entrez id for all genes
    entrez_map = {
        **dict(zip(lv["gene_symbol"], lv["gene_entrez_id"])),
        **dict(zip(deseq2["gene_symbol"], deseq2["gene_entrez_id"])),
    }
    return gene_ids, symbol_map, entrez_map


# this prepares the tpm matrix for clustering with transformations.  tpm is good for between-sample comparisons.
def prep_matrix_for_clustering(
    tpm_matrix,
    gene_ids,
    symbol_map,
):
    tpm_matrix = tpm_matrix.reset_index()
    deg_reads = tpm_matrix[
        tpm_matrix["gene_id_mapped"].isin([int(x) for x in gene_ids])
    ]
    deg_reads["symbol"] = deg_reads["gene_id_mapped"].map(symbol_map)
    deg_reads = deg_reads.set_index("symbol")
    deg_reads = deg_reads.drop(columns="gene_id_mapped")
    return deg_reads


# this performs hierarchical clustering
def hierarchical_clustering(deg_reads, exp_design):
    test = deg_reads.T.merge(exp_design, how="left", left_index=True, right_index=True)
    test = test.rename(columns={"act_demented": "Condition"})
    condition = test.pop("Condition")

    colors = dict(
        zip(
            condition.unique(),
            sns.color_palette(
                "Spectral",
                2,
            ),
        )
    )  # https://seaborn.pydata.org/generated/seaborn.clustermap.html
    row_colors = condition.map(colors)

    hier = sns.clustermap(
        test,
        method="ward",
        cmap="Spectral",
        row_colors=row_colors,
        z_score=1,
        metric="euclidean",
        dendrogram_ratio=0.15,
        figsize=(12, 10),
        yticklabels=False,
        cbar_pos=(0.03, 0.95, 0.2, 0.02),
        cbar_kws={
            "orientation": "horizontal"
        },  # https://stackoverflow.com/questions/67040271/reduce-space-between-dendrogram-and-color-row-in-seaborn-clustermap
    )
    for label in condition.unique():
        hier.ax_row_dendrogram.bar(0, 0, color=colors[label], label=label, linewidth=0)
    hier.ax_row_dendrogram.legend(
        title="Condition",
        loc="center",
        bbox_transform=gcf().transFigure,
        bbox_to_anchor=(0.95, 0.95),
    )  # https://stackoverflow.com/questions/27988846/how-to-express-classes-on-the-axis-of-a-heatmap-in-seaborn
    hier.ax_heatmap.set_xlabel("Genes")
    hier.ax_col_dendrogram.set_title("Hierarchical Clustering - Gene Expression")
    hier.cax.set_title("TPM Normalized", fontdict={"size": 8})
    hier.cax.tick_params(labelsize=8)

    l1 = hier.ax_heatmap.set_ylabel(
        "Samples"
    )  # https://stackoverflow.com/questions/46234158/how-to-remove-x-and-y-axis-labels-in-a-clustermap
    plt.show()
    plt.figure(figsize=(6, 15))
    cls = ward(deg_reads.T)
    den = dendrogram(cls, orientation="left", p=4, truncate_mode="level")
    plt.title("Hierarchical Clustering - Gene Expression")
    plt.show()


# this scales the matrix for future clustering tasks
def scale_deg_reads(deg_reads):
    deg_reads_t = deg_reads.T
    X_arr = deg_reads_t.to_numpy()
    scale_x = StandardScaler()
    x = scale_x.fit_transform(X_arr)
    return x


# this performs k-means clustering
def kmeans_clustering(x):
    lst = []
    print("kmeans evaluation")

    for i in range(2, 20):
        kmeans = KMeans(
            n_clusters=i,
            init="k-means++",
            max_iter=200,
            n_init=1,
            random_state=42,
        )
        c = kmeans.fit(x)
        print(
            "cluster {}: davies-bouldin: {} calinski harabasz: {} silhouette: {}".format(
                i,
                sklearn.metrics.davies_bouldin_score(x, kmeans.labels_),
                sklearn.metrics.calinski_harabasz_score(x, kmeans.labels_),
                sklearn.metrics.silhouette_score(x, kmeans.labels_),
            )
        )
        lst.append((c.inertia_))
    plt.plot([x for x in range(2, 20)], lst)
    plt.xticks([x for x in range(2, 20)])
    plt.show()
    kmeans = KMeans(
        n_clusters=2, init="k-means++", max_iter=100, n_init=1, random_state=42
    )  # code from 543
    labels_kmeans = kmeans.fit_predict(x)
    return labels_kmeans


# this performs kmedoids clustering
def kmedoids_clustering(x):
    lst = []
    print("kmedoids evaluation")

    for i in range(2, 20):
        km = kmedoids.KMedoids(i, method="fasterpam", metric="euclidean")
        c = km.fit(x)
        print(
            "cluster {}: davies-bouldin: {} calinski harabasz: {} silhouette: {}".format(
                i,
                sklearn.metrics.davies_bouldin_score(x, km.labels_),
                sklearn.metrics.calinski_harabasz_score(x, km.labels_),
                sklearn.metrics.silhouette_score(x, km.labels_),
            )
        )
        lst.append((c.inertia_))
    print("kmedoids evaluation")
    plt.plot([x for x in range(2, 20)], lst)
    plt.xticks([x for x in range(2, 20)])
    plt.show()
    km = kmedoids.KMedoids(2, method="fasterpam", metric="euclidean")
    labels_kmed = km.fit_predict(x)
    return labels_kmed


# this performs hdbscan clustering
def hdbscan_clustering(x):
    hdb = HDBSCAN(min_cluster_size=3)
    hdb.fit(x)
    db_labels = hdb.labels_
    return db_labels


# this performs pca
def pca_analysis(x):
    pca = PCA(2)
    pca_res = pca.fit_transform(x)
    return pca_res


# this graphs the pca and colors the samples based on cluster
def cluster_pca_graphs(n_clusters, labels, pca_res, title):
    for i in range(n_clusters):
        if i == 1:
            x = "dodgerblue"
        else:
            x = "yellowgreen"
        plt.scatter(
            pca_res[labels == i, 0], pca_res[labels == i, 1], label=i, c=x
        )  # https://medium.com/analytics-vidhya/implementation-of-principal-component-analysis-pca-in-k-means-clustering-b4bc0aa79cb6
    plt.legend()
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.show()


# this creates a series of exploratory graphs to try to understand the clustering
def cluster_demo_graphs(
    labels_kmeans,
    labels_kmed,
    db_labels,
    deg_reads,
    tbidata,
    donor,
):
    label_merge = tbidata.merge(donor, how="left", on="donor_id")
    label_merge = label_merge[
        [
            "structure_name",
            "rnaseq_profile_id",
            "age",
            "sex",
            "apo_e4_allele",
            "education_years",
            "age_at_first_tbi",
            "longest_loc_duration",
            "cerad",
            "num_tbi_w_loc",
            "dsm_iv_clinical_diagnosis",
            "control_set",
            "nincds_arda_diagnosis",
            "ever_tbi_w_loc",
            "race",
            "hispanic",
            "act_demented",
            "braak",
            "nia_reagan",
        ]
    ]
    clustering = deg_reads.T
    clustering["labels_kmeans"] = [str(x) for x in labels_kmeans]
    clustering["labels_db"] = [str(x) for x in db_labels]
    clustering["labels_kmed"] = [str(x) for x in labels_kmed]
    clusters_df = clustering.merge(
        label_merge, how="left", left_index=True, right_on="rnaseq_profile_id"
    )
    clusters_df["braak"] = clusters_df["braak"].astype(int)
    for i in [
        "sex",
        "dsm_iv_clinical_diagnosis",
        "nincds_arda_diagnosis",
        "control_set",
        "ever_tbi_w_loc",
        "age",
        "structure_name",
        "act_demented",
    ]:
        for y in ["labels_kmed", "labels_kmeans", "labels_db"]:

            gb = clusters_df.groupby(by=[i, y]).count().reset_index()
            fig = px.bar(
                gb, x=y, y="SST", color=i, title="{} Comparison for Clusters".format(i)
            )
            fig.show()

    for i in ["braak", "nia_reagan"]:
        for y in ["labels_kmed", "labels_kmeans", "labels_db"]:

            gb = clusters_df.groupby(by=[y])[i].mean().reset_index()
            fig = px.bar(gb, x=y, y=i, title="{} Comparison for Clusters".format(i))
            fig.show()
    return clustering, clusters_df


# this function creates the volcano plot for limma-voom results
def lv_volc_plot(lv_res):
    # Camaron's code inspired by Hemtools - https://hemtools.readthedocs.io/en/latest/content/Bioinformatics_Core_Competencies/Volcanoplot.html
    plt.figure(figsize=(8, 8))
    plt.scatter(
        x=lv_res["logFC"],
        y=lv_res["adj.P.Val"].apply(lambda x: -np.log10(x)),
        s=1,
        label="Not significant",
        color="gray",
    )

    # highlight down- or up- regulated genes
    down = lv_res[(lv_res["logFC"] < -0.5) & (lv_res["adj.P.Val"] <= 0.05)]
    up = lv_res[(lv_res["logFC"] >= 0.5) & (lv_res["adj.P.Val"] <= 0.05)]

    plt.scatter(
        x=down["logFC"],
        y=down["adj.P.Val"].apply(lambda x: -np.log10(x)),
        s=3,
        label="Down-regulated",
        color="blue",
    )
    plt.scatter(
        x=up["logFC"],
        y=up["adj.P.Val"].apply(lambda x: -np.log10(x)),
        s=3,
        label="Up-regulated",
        color="red",
    )

    plt.xlabel("log2FoldChange")
    plt.ylabel("-logFDR")
    plt.axvline(-0.5, color="grey", linestyle="--")
    plt.axvline(0.5, color="grey", linestyle="--")
    plt.axhline(1.3, color="grey", linestyle="--")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.legend()
    plt.title("Volcano Plot of Limma-Voom Results")
    plt.show()


# this creates plots to show the difference in gene expression between clusters
def gene_expression_diff_graph(labels, clustering, title):
    gene_cols = [
        "LOC105378594",
        "TSNAX",
        "CALM2",
        "ERLEC1",
        "HIGD1A",
        "IQCJ-SCHIP1",
        "SST",
        "DOK7",
        "SMIM14",
        "SCOC",
        "KIAA0408",
        "MICALL2",
        "TNRC18",
        "TAC1",
        "LOC728433",
        "LOC100113421",
        "COL27A1",
        "LOC100130992",
        "GAD2",
        "BCL9L",
        "SLC6A12",
        "ATN1",
        "LOC642969",
        "ACTR6",
        "POLE",
        "KIF26A",
        "RCN2",
        "CAPN15",
        "LOC101927793",
        "LOC102724632",
        "LOC105371409",
        "LOC105372247",
        "CIC",
        "BCAM",
        "FAM65C",
        "PPEF1",
        "BEX4",
    ]
    symbol_graph = pd.melt(
        clustering.groupby(labels)[gene_cols].mean().T.reset_index(),
        id_vars="symbol",
        value_vars=[0, 1],
    )
    diff = clustering.groupby(labels)[gene_cols].mean().T.reset_index()
    diff["ratio"] = diff[0] / diff[1]
    plt.figure(figsize=(15, 6))

    ratio_plot = sns.barplot(data=diff, x="symbol", y="ratio", color="dodgerblue")
    plt.title(title)
    plt.xticks(rotation="vertical")
    plt.xlabel("Genes")
    plt.ylabel("Average TPM Ratio")
    plt.show()


# this shoes difference in samples with or without dementia between clusters
def dementia_cluster_plot(clusters_df, labels, title):
    lst = []
    for i in ["0", "1"]:
        dementia_percent = (
            len(
                clusters_df[
                    (clusters_df["act_demented"] == "Dementia")
                    & (clusters_df[labels] == i)
                ]
            )
            / len(clusters_df[(clusters_df[labels] == i)])
            * 100
        )

        no_dementia_percent = (
            len(
                clusters_df[
                    (clusters_df["act_demented"] == "No Dementia")
                    & (clusters_df[labels] == i)
                ]
            )
            / len(clusters_df[(clusters_df[labels] == i)])
            * 100
        )
        lst.append((i, no_dementia_percent, "No Dementia"))
        lst.append((i, dementia_percent, "Dementia"))

    dementia_percent_df = pd.DataFrame(lst, columns=["Cluster", "Value", "Status"])
    dementia_percent_df = dementia_percent_df.pivot(
        index="Cluster", columns="Status", values="Value"
    )
    dementia_plot = dementia_percent_df.plot(
        kind="bar", stacked=True, color=["dodgerblue", "yellowgreen"]
    )  # https://www.statology.org/seaborn-stacked-bar-plot/
    plt.ylabel("% of Total")
    plt.gca().yaxis.set_major_formatter(
        mtick.PercentFormatter()
    )  # https://stackoverflow.com/questions/31357611/format-y-axis-as-percent
    plt.title(title)
    return dementia_percent_df


# this performs enrichment analysis
def enrichment_analysis(
    lv_genes_merged,
):
    enr = gp.enrichr(
        gene_list=[str(x) for x in list(list(lv_genes_merged["gene_symbol"]))],
        gene_sets=[
            "MSigDB_Hallmark_2020",
            "KEGG_2021_Human",
            "GO_Biological_Process_2018",
            "WikiPathways_2016",
            "GO_Molecular_Function_2015",
            "GO_Cellular_Component_2015",
            "Human_Gene_Atlas",
            "Disease_Signatures_from_GEO_up_2014",
            "Disease_Signatures_from_GEO_down_2014",
        ],
        organism="human",
        outdir=None,
    )
    # categorical scatterplot
    ax = dotplot(
        enr.results,
        column="Adjusted P-value",
        x="Gene_set",
        size=10,
        top_term=5,
        figsize=(3, 5),
        title="Enrichment Analysis",
        xticklabels_rot=45,
        show_ring=True,
        marker="o",
    )
    # https://gseapy.readthedocs.io/en/latest/gseapy_example.html
    return enr
