import cPickle as pickle

def sampleSplit(samples, filename):
  result = {}
  for genere in samples.keys():
    mylists = {}
    mylists[0] = samples[genere][0]
    mylists[1] = samples[genere][1]
    mylists[2] = samples[genere][2]
    mylists[3] = samples[genere][3]
    mylists[4] = samples[genere][4]
    mylists[5] = samples[genere][5]
    mylists[6] = samples[genere][6]
    mylists[7] = samples[genere][7]
    result[genere] = mylists
  filename = "./data/" + filename
  o = open(filename, 'w')
  pickle.dump(result, o)
  o.close()

'''Read in data'''
samples = pickle.load( open( "../serverTest2/MusicClassification/data/data.in", "rb" ) )
sampleSplit(samples, "data_small8II.in")