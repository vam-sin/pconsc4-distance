import h5py

test_file_name = "../Datasets/PconsC4-data/data/test_plm-gdca-phy-ss-rsa-eff-ali-mi_new.h5"
f_test = h5py.File(test_file_name, 'r')

key_lst = list(f_test['gdca'].keys())

strpdb = ''
pdb = []
chain = []

for i in range(len(key_lst)):
	key_lst[i] = key_lst[i].replace('.hhE0', '')
	pdbid = key_lst[i][0:4]
	ch = key_lst[i][-1]
	strpdb += pdbid
	strpdb += ', '
	pdb.append(pdbid)
	chain.append(ch)

print(strpdb)