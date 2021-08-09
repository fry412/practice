# 数据：
- 主要涉及的数据集是：1）General QA数据集（SQuAD）和BioASQ Challenge数据集。
- BioASQ的数据可以从[官网](http://bioasq.org/) 下载，需要先注册账号。之后如果参加比赛也是在这个平台。这部分数据<font color=red>**不在**</font>本folder里，如果需要请注册下载。
- data中的实验数据是在韩国实验室[DMIS-LAB](https://github.com/dmis-lab) enrich之后的数据```data/originalDMIS_LAB```的基础上进行的POS和NER特征抽取。
# code
## mainCode
- 这个版本的code还是基于最早的tensorflow 1.x的BERT做的。2020年中旬有pytorch的BioBERT版本，可以自己找一下。
- paper里主要的模型直接加在了```run_factoid_pos_ner.py```里。
## evaluationTools
- 这里也是用到了DMIS-LAB的处理脚本```code/evaluationTools/biocodes```，要先把BERT prediction的结果trnsfer成BioASQ的官方格式。
- 附上三个脚本文件，用于train、predict模型以及批量使用BioASQ官方的evaluation metric对结果进行评估。
## otherRemarks
- ```caseAnalysis_humanevaluation.txt```： 这是我人工对6、7、8b的test结果的一个分析，里面有很多不可回答的问题，可以参考，之后可以尝试引入生成式模型（paper里有写）
- ```Bio结论.xlsx```： 实验结果以及POS、NER feature的选择
# relatedPapers
2020年BioQA的相关文章
# others
其他细节和实验implementation可以参考[Github](https://github.com/xugezheng/BioQAExternalFeatures) 上的内容