import pandas

cands = pandas.read_csv("../data/parties/list_of_candidates_final_hand_edited.csv", sep=',')

cands['uidname']=cands['leader'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
cands['uidname']=cands['uidname'].str.lower()
cands['uidname']=cands['uidname'].str.replace(" ", "_")
cands['uidname']=cands['uidname'].str.replace(".", "")
cands['uidname']=cands['uidname'].str.replace("'", "")
cands['uidname']=cands['uidname'].str.replace("-", "_")
cands['uidname']=cands['uidname'].str.replace(",", "")

### remove minor candidates
cands=cands[(cands.vote_share >= 5) & (cands.vote_share <= 100)]
cands=cands.dropna(subset=['uidname'])

print(cands['uidname'])

cands.to_csv("../data/parties/list_of_candidates_final.csv", sep=',')
cands.to_csv("../output/list_of_candidates_final.csv", sep=',')
