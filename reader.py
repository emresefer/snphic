import sys
import os
import math
import gzip
import numpy as np
import random

def readFreqMatrixSize(freqpath):
    """reads freq matrix size without reading data
    Args:
       freqpath:
    Returns:
       nodecount:
    """
    if freqpath.endswith(".gz"):
       compress = True
    nodecount = 0
    if compress:
       with gzip.open(freqpath,"r") as infile:
          for line in infile:
              nodecount += 1
    else:
       with open(freqpath,"r") as infile:
          for line in infile:
              nodecount += 1       
    return nodecount   


def readFreqFile(freqfile):
    """reads input matrix file in gz format
    Args:
       freqfile:
    Returns:
       freqmat: numpy 2-d array
       nodenames: list
       in2pos: index to genome start end locations~(list)
    """    
    nodenames,in2pos = [], []
    compress = False
    if freqfile.endswith(".gz"):
       compress = True
    if compress:
       with gzip.open(freqfile,"r") as infile:
            for line in infile:
                node,start,end = line.rstrip().split("\t")[0:3]
                nodenames.append(node)
                in2pos.append((int(start),int(end)))    
    else:
       with open(freqfile,"r") as infile:
            for line in infile:
                node,start,end = line.rstrip().split("\t")[0:3]
                nodenames.append(node)
                in2pos.append((int(start),int(end)))             
    freqmat = np.zeros((len(nodenames),len(nodenames)),dtype=np.float)
    index = 0 
    if compress:      
       with gzip.open(freqfile,"r") as infile:
            for line in infile:
                parts = line.rstrip().split("\t")
                for index2 in xrange(len(nodenames)):
                    freqmat[index,index2] = float(parts[3+index2])
                index+=1
    else:
       with open(freqfile,"r") as infile:
            for line in infile:
                parts = line.rstrip().split("\t")
                for index2 in xrange(len(nodenames)):
                    freqmat[index,index2] = float(parts[3+index2])
                index+=1         
    return freqmat,nodenames,in2pos
            
def readDomainFile(domfile):
    """reads domain file
    Args:
       domfile:
    Returns:
       domainlist: first index 1 not 0.
    """
    domainlist, doms = [], []
    with open(domfile,"r") as infile:
        for line in infile:
            line = line.rstrip()
            if line == "":
               domainlist.append(list(doms))
               doms = []
               continue
            splitted = line.split(",")
            doms.append((int(splitted[0]),int(splitted[1])))
    domainlist.append(list(doms))        
    return domainlist


def addEmptyClusters(domains,allnodes,TESTMODE=False):
    """adds empty clusters to make a full partition
    Args:
       domains: TADs
       allnodes: all possible segments
       TESTMODE: run tests
    Returns:
       fullpart: full partition of nodes
    """        
    part1 = sorted([range(start,end+1) for start,end in domains], key=lambda item: item[0])
    curin,part2 = 1,[]
    for clust in part1:
        start,end = min(clust),max(clust)
        if curin <= start:
           part2.append(range(curin,start))
        curin = end+1
    if curin <= len(allnodes):
       part2.append(range(curin,len(allnodes)+1))
    if TESTMODE:   
       for clust1 in part1:
           for clust2 in part2:
               assert len(set(clust1) & set(clust2)) == 0
    part1.extend(part2)
    part1 = [part for part in part1 if len(part)>0]
    if TESTMODE:  
       for ind1 in xrange(len(part1)):
           range1 = set(part1[ind1]) 
           for ind2 in xrange(ind1+1,len(part1)):
               range2 = set(part1[ind2])
               assert len(range1 & range2) == 0
       assert len(set(node for part in part1 for node in part) ^ set(allnodes)) == 0        
    return part1


def shuffleTAD(chro2doms,nodemap,TESTMODE=False):
    """shuffle TADs by preserving length distribution
    Args:
       chro2doms: dictionary of chromosome to TADs
       nodemap: dictionary of chromosome to number of segments
       TESTMODE: runtest flag
    Returns:
       outdoms: shuffled domains
    """
    outdoms = {}
    for chro,domains in chro2doms.items():
        len2dist = {}
        for start,end in domains:
            len2dist.setdefault(end-start+1,0)
            len2dist[end-start+1] += 1
        nodecount = nodemap[chro]
        allnodes = range(1,nodecount+1)
        truepart = addEmptyClusters(domains,allnodes,TESTMODE)
        seennodes = set(item for clu in truepart for item in clu)
        assert len(seennodes ^ set(allnodes)) == 0
        trueclust = sorted(truepart)
        lens = [len(clust) for clust in trueclust]
        random.shuffle(lens)
        locs = [0]
        for ind in xrange(0,len(lens)-1):
            locs.append(lens[ind]+locs[ind])
        if len(locs) > 1:    
           locs = locs[1:]
        curclust = [allnodes[0:locs[0]]]
        for ind in xrange(len(locs)-1):
            curclust.append(allnodes[locs[ind]:locs[ind+1]])
        curclust.append(allnodes[locs[-1]:nodecount])
        curclust = [clu for clu in curclust if len(clu) != 0]
        putcurclust = set((clu[0],clu[-1]) for clu in curclust)
        len2info,len2doms = {}, {}
        for start,end in putcurclust:
            len2info.setdefault(end-start+1,[])
            len2info[end-start+1].append((start,end))
        for mylen in len2info.keys():
            if not len2dist.has_key(mylen):
               continue
            blocks = len2info[mylen]
            random.shuffle(blocks)
            len2doms[mylen] = set(blocks[0:len2dist[mylen]])
        putdoms = [block for mylen in len2doms.keys() for block in len2doms[mylen]]
        outdoms[chro] = list(putdoms)
        if TESTMODE:
           tsum,tsum2,tsum3 = sum([end-start+1 for start,end in putcurclust]), sum([end-start+1 for start,end in putdoms]), sum([end-start+1 for start,end in domains])
           assert tsum == nodecount and tsum2 == tsum3
           seen1 = set(item for clu in curclust for item in clu)
           tseen1 = [item for clu in curclust for item in clu]
           seen2 = set(allnodes)
           seen3 = set(item for clu in trueclust for item in clu)
           assert len(seen1 ^ seen2) == 0 and len(seen1 ^ seen3) == 0 and len(seen1) == len(tseen1)
    return outdoms

freqfile = "normalized/1.freq.gz"
#freqmat, nodenames, in2pos = readFreqFile(freqfile)
domfile = "tADs/1.domains"
domains = readDomainFile(domfile)
print domains
chro2doms = {1: list(domains[0])}
shufdoms = shuffleTAD(chro2doms,nodemap,TESTMODE=True)
print shufdoms
