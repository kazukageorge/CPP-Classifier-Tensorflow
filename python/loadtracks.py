import scipy.io
import pandas as pd
from itertools import chain
from itertools import groupby
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datetime import datetime

print("\nfunc lt\n")

def pullMatData(cellNum):
    #Load .mat tracks file
    mat = scipy.io.loadmat('KangminData/Cell' + str(cellNum) + '_1s/TagRFP/Tracking/ProcessedTracks.mat')
    tracks = mat['tracks'][0,:]
    #print("PULLING MAT DATA FOR CELL " + str(cellNum) + ":")

    #Initialize DF
    df = pd.DataFrame(columns=['frame', 'lifetime', 'max_intensity', 'background', 'totaldisp', 'max_msd', 'catldy', 'aux', 'avg_rise', 'avg_dec', 'risevsdec', 'avg_mom_rise', 'avg_mom_dec', 'risevsdec_mom'])

    #Initialize for Histogram:
    pvalList = list()

    #Populate each track
    for i in range(len(tracks)):
        #Pull Individual Values:

        #Intensity (A)
        intensity = tracks[i][4]
        intensity_t1 = list(intensity[0])
        intensity_t2 = list(intensity[1])
        intensity_max = max(intensity_t1)
        lifetime = len(intensity_t1)
        max_index = intensity_t1.index(intensity_max)

        #Background(c)
        bkground = tracks[i][5]
        bkground_t1 = list(bkground[0])
        bkground_t2 = list(bkground[1])
        bkground_scal = bkground_t1[max_index]

        #Category(catldx)
        catldx = list(chain.from_iterable(tracks[i][33]))
        catldx_scal = catldx[0]

        #Pvals(pvals)
        pvals = tracks[i][12]
        pvals_t2 = list(pvals[1])
        
        #Frame(f)
        frame = list(chain.from_iterable(tracks[i][1]))
        frame_scal = frame[max_index]

        #Start Buffer(startBuffer)
        startBuffer = list(chain.from_iterable(tracks[i][24]))

        #End Buffer(endBuffer)
        endBuffer = list(chain.from_iterable(tracks[i][25]))

        #Motion Analysis:
        motion_analysis = list(chain.from_iterable(tracks[i][26]))
        if not motion_analysis: #If motion analysis is empty...
            tdisp_scal = None
            msd_max = None
        else:
            tdisp = list(chain.from_iterable(motion_analysis[0][0]))
            tdisp_scal = tdisp[0]
            msd = list(chain.from_iterable(motion_analysis[0][1]))
            msd_max = max(msd)
        #endif

        #Check if Aux+ or Aux-
        isAux = False
        pval_cutoff = .005
        #Look for 5+ Consecutive True's
        L = ((pvals_t2[i] <= pval_cutoff and pvals_t2[i + 1] <= pval_cutoff and pvals_t2[i + 2] <= pval_cutoff) for i in range(len(pvals_t2)-2))
        isAux = any(L)
        grouped_L = [(k, sum(1 for i in g)) for k,g in groupby(L)]
        for item in grouped_L:
            if item:
                if item[0] == True:
                    pvalList.append(item[1])
        if isAux:
            aux = 1
        else:
            aux = 0
        #endif

        #Generate Full Intensity for Min / Max Rise / Decay
        if startBuffer and endBuffer:
            intensity_list = list(startBuffer[0][3][0]) + intensity_t1 + list(endBuffer[0][3][0])
            start_index = 1
            end_index = len(intensity_list)-1
        else:
            intensity_list = [0] + intensity_t1 + [0]
            start_index = intensity_list.index(intensity_t1[0])
            end_index = intensity_list.index(intensity_t1[-1])
        rise_slope = list()
        decay_slope = list()
        total_slope = list()
        for j in range(start_index, end_index + 1):
            slope = intensity_list[j] - intensity_list[j - 1]
            if slope >= 0:
                rise_slope.append(slope)
            else:
                decay_slope.append(slope)
            total_slope.append(slope)
        if len(rise_slope) == 0 and len(decay_slope) == 0:
            avgRise = 0
            avgDecay = 0
            riseVsDecay = 0
            avgRiseMom = 0
            avgDecMom = 0
            riseVsDecayMom = 0
        elif len(rise_slope) == 0:   
            avgRise = 0
            avgDecay = sum(decay_slope) / len(decay_slope)
            riseVsDecay = 0
            avgRiseMom = 0
            avgDecMom = 0
            riseVsDecayMom = 0
        elif len(decay_slope) == 0:   
            avgRise = sum(rise_slope) / len(rise_slope)
            avgDecay = 0
            riseVsDecay = 0
            avgRiseMom = 0
            avgDecMom = 0
            riseVsDecayMom = 0
        else:   
            avgRise = sum(rise_slope) / len(rise_slope)
            avgDecay = sum(decay_slope) / len(decay_slope)
            riseVsDecay = len(rise_slope) / len(decay_slope)

            #Get Moments
            rise_slope_mom = list()
            decay_slope_mom = list()
            start_index = 1
            end_index = len(total_slope)-1
            for j in range(start_index, end_index + 1):
                moment = total_slope[j] - total_slope[j - 1]
                if moment >= 0:
                    rise_slope_mom.append(moment)
                else:
                    decay_slope_mom.append(moment)
            if len(rise_slope_mom) == 0 and len(decay_slope_mom) == 0:
                avgRiseMom = 0
                avgDecMom = 0
                riseVsDecayMom = 0
            elif len(rise_slope_mom) == 0:   
                avgRiseMom = 0
                avgDecMom = 0
                riseVsDecayMom = 0
            elif len(decay_slope_mom) == 0:   
                avgRiseMom = 0
                avgDecMom = 0
                riseVsDecayMom = 0
            else:   
                avgRiseMom = sum(rise_slope_mom) / len(rise_slope_mom)
                avgDecayMom = sum(decay_slope_mom) / len(decay_slope_mom)
                riseVsDecayMom = len(rise_slope_mom) / len(decay_slope_mom)
            #endif
        #endif

        #Create Row
        row = [frame_scal, lifetime, intensity_max, bkground_scal, tdisp_scal, msd_max, catldx_scal, aux, avgRise, avgDecay, riseVsDecay, avgRiseMom, avgDecayMom, riseVsDecayMom]
        
        # Add to DF
        df.loc[i] = row

    #End For Loop

    #Final Checks:
    df = df[df['max_msd'].notnull()]

    return (df, pvalList)

def pullTrackData(cellNum):
    #Load .mat tracks file
    mat = scipy.io.loadmat('KangminData/Cell' + str(cellNum) + '_1s/TagRFP/Tracking/ProcessedTracks.mat')
    tracks = mat['tracks'][0,:]
    #print("PULLING MAT DATA FOR CELL " + str(cellNum) + ":")

    #Initialize DF
    df = pd.DataFrame(columns=['frame', 'lifetime', 'max_intensity', 'background', 'totaldisp', 'max_msd', 'catldy', 'aux', 'avg_rise', 'avg_dec', 'risevsdec', 'avg_mom_rise', 'avg_mom_dec', 'risevsdec_mom'])
    df_tot = pd.DataFrame(columns=['ap2', 'aux'])
    #Initialize for Histogram:
    pvalList = list()

    #Populate each track
    for i in range(len(tracks)):
        #Pull Individual Values:

        #Intensity (A)
        intensity = tracks[i][4]
        intensity_t1 = list(intensity[0])
        intensity_t2 = list(intensity[1])
        intensity_max = max(intensity_t1)
        lifetime = len(intensity_t1)
        max_index = intensity_t1.index(intensity_max)

        #Background(c)
        bkground = tracks[i][5]
        bkground_t1 = list(bkground[0])
        bkground_t2 = list(bkground[1])
        bkground_scal = bkground_t1[max_index]

        #Category(catldx)
        catldx = list(chain.from_iterable(tracks[i][33]))
        catldx_scal = catldx[0]

        #Pvals(pvals)
        pvals = tracks[i][12]
        pvals_t2 = list(pvals[1])
        
        #Frame(f)
        frame = list(chain.from_iterable(tracks[i][1]))
        frame_scal = frame[max_index]

        #Start Buffer(startBuffer)
        startBuffer = list(chain.from_iterable(tracks[i][24]))

        #End Buffer(endBuffer)
        endBuffer = list(chain.from_iterable(tracks[i][25]))

        #Motion Analysis:
        motion_analysis = list(chain.from_iterable(tracks[i][26]))
        if not motion_analysis: #If motion analysis is empty...
            tdisp_scal = None
            msd_max = None
        else:
            tdisp = list(chain.from_iterable(motion_analysis[0][0]))
            tdisp_scal = tdisp[0]
            msd = list(chain.from_iterable(motion_analysis[0][1]))
            msd_max = max(msd)
        #endif

        #Check if Aux+ or Aux-
        isAux = False
        pval_cutoff = .005
        #Look for 5+ Consecutive True's
        L = ((pvals_t2[i] <= pval_cutoff and pvals_t2[i + 1] <= pval_cutoff and pvals_t2[i + 2] <= pval_cutoff) for i in range(len(pvals_t2)-2))
        isAux = any(L)
        grouped_L = [(k, sum(1 for i in g)) for k,g in groupby(L)]
        #for item in grouped_L:
        #    if item:
        #        if item[0] == True:
        #            pvalList.append(item[1])
        if isAux:
            aux = 1
        else:
            aux = 0
        #endif

        #Generate Full Intensity for Min / Max Rise / Decay
        if startBuffer and endBuffer:
            intensity_list = list(startBuffer[0][3][0]) + intensity_t1 + list(endBuffer[0][3][0])
            start_index = 1
            end_index = len(intensity_list)-1
        else:
            intensity_list = [0] + intensity_t1 + [0]
            start_index = intensity_list.index(intensity_t1[0])
            end_index = intensity_list.index(intensity_t1[-1])
        rise_slope = list()
        decay_slope = list()
        total_slope = list()
        for j in range(start_index, end_index + 1):
            slope = intensity_list[j] - intensity_list[j - 1]
            if slope >= 0:
                rise_slope.append(slope)
            else:
                decay_slope.append(slope)
            total_slope.append(slope)
        if len(rise_slope) == 0 and len(decay_slope) == 0:
            avgRise = 0
            avgDecay = 0
            riseVsDecay = 0
            avgRiseMom = 0
            avgDecMom = 0
            riseVsDecayMom = 0
        elif len(rise_slope) == 0:   
            avgRise = 0
            avgDecay = sum(decay_slope) / len(decay_slope)
            riseVsDecay = 0
            avgRiseMom = 0
            avgDecMom = 0
            riseVsDecayMom = 0
        elif len(decay_slope) == 0:   
            avgRise = sum(rise_slope) / len(rise_slope)
            avgDecay = 0
            riseVsDecay = 0
            avgRiseMom = 0
            avgDecMom = 0
            riseVsDecayMom = 0
        else:   
            avgRise = sum(rise_slope) / len(rise_slope)
            avgDecay = sum(decay_slope) / len(decay_slope)
            riseVsDecay = len(rise_slope) / len(decay_slope)

            #Get Moments
            rise_slope_mom = list()
            decay_slope_mom = list()
            start_index = 1
            end_index = len(total_slope)-1
            for j in range(start_index, end_index + 1):
                moment = total_slope[j] - total_slope[j - 1]
                if moment >= 0:
                    rise_slope_mom.append(moment)
                else:
                    decay_slope_mom.append(moment)
            if len(rise_slope_mom) == 0 and len(decay_slope_mom) == 0:
                avgRiseMom = 0
                avgDecMom = 0
                riseVsDecayMom = 0
            elif len(rise_slope_mom) == 0:   
                avgRiseMom = 0
                avgDecMom = 0
                riseVsDecayMom = 0
            elif len(decay_slope_mom) == 0:   
                avgRiseMom = 0
                avgDecMom = 0
                riseVsDecayMom = 0
            else:   
                avgRiseMom = sum(rise_slope_mom) / len(rise_slope_mom)
                avgDecayMom = sum(decay_slope_mom) / len(decay_slope_mom)
                riseVsDecayMom = len(rise_slope_mom) / len(decay_slope_mom)
            #endif
        #endif

        #Create Row
        row = [frame_scal, lifetime, intensity_max, bkground_scal, tdisp_scal, msd_max, catldx_scal, aux, avgRise, avgDecay, riseVsDecay, avgRiseMom, avgDecayMom, riseVsDecayMom]
        
        # Add to DF
        df_int = pd.DataFrame(data = {'ap2': intensity_t1, 'aux': intensity_t2})
        #Conditions:
        if(1 <= catldx_scal <= 4 and avgDecay < 0):
            df_tot = pd.concat([df_tot, df_int], ignore_index = True)

    #End For Loop

    return (df_tot)
    
