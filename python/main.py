
'''
1. Show video of the tracks over time, 1. cl, 2. au, 3. merge
    circle the PSF
2. show plots of individual, randomly selected, save one as above

preprocess



'''
import os, sys, getopt
import math
import numpy as np
import pandas as pd
import scipy.io
import pandas as pd
from itertools import chain
from IPython import display
from tensorflow.python.data import Dataset

from itertools import groupby
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datetime import datetime

import tensorflow as tf

def get_args(argv=None):
    # ADDD PLOTS
    if argv == None:
        sys.exit(2)
    verbose = False
    inputData = None

    try:
        opts, args = getopt.getopt(argv, "hvi:",
                                   [
                                       "help",
                                       "verbose",
                                       "input",
                                   ])
    except getopt.GetoptError as err:
        print('Invalid input arguments..')
        print('Please type \npython main.py --help')
        print(str(err))
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('\
                -h --help:                      help command  \n\
                -v --verbose:                   verbose option\n\
                -i --input <directory>:         input directory to "~/InputData \n\
                    ')

            sys.exit()
        elif opt in ("-v", "--verbose"):
            check_common_feat = True
        elif opt in ("-i", "--input"):
            inputData = arg


    print('-'*100)
    print('Input Arguments: \n')
    # print('VERBOSE:                  {}'.format(verbose))
    print('INPUT_DATA_DIRECTORY:     {}'.format(inputData))
    # print('-'*100)

    if not inputData:
        print('Please type \npython main.py --help')

    return verbose, inputData

def get_inputData(parentFolderPath,verb):
    # get input Data
    print("-"*100)
    print('Organizing .mat files of molecule tracking data in to DataFrame\n')
    print('Dataframe querried from clathrin info, intensity spike from auxilin appended to df ')
    # if dataframe already organized, load the data
    if os.path.exists(parentFolderPath+os.path.sep+'dataframe'+os.path.sep+"aux.csv"):
        df = pd.read_csv(parentFolderPath+os.path.sep+'dataframe'+os.path.sep+"aux.csv")
        df = df.set_index('trackNum')
        print("df loaded from {}".format(parentFolderPath+os.path.sep+'dataframe'+os.path.sep+"aux.csv"))
        return df

    pvalCutOff = 0.05
    numConsecPVal=3
    mustBeSecondHalf =True
    showPrints = True
    showBoxplot = checkLifetime =False
    cols = ['trackNum', 'frame', 'lifetime', 'max_intensity', 'background', 'totaldisp', 'max_msd', 'catIdx', 'aux', 'avg_rise', 'avg_dec', 'risevsdec', 'avg_mom_rise', 'avg_mom_dec', 'risevsdec_mom']
    print('Importing data to dataframe with columns: ', end =" ")
    for feat in cols:
        print(feat, end=", ")
    print('')
    df = pd.DataFrame(columns=cols)
    # df.set_index('trackNum')

    for cellNum in range(1,9):
        df_temp = pd.DataFrame(
            columns=['trackNum', 'frame', 'lifetime', 'max_intensity', 'background', 'totaldisp', 'max_msd', 'catIdx',
                     'aux', 'avg_rise', 'avg_dec', 'risevsdec', 'avg_mom_rise', 'avg_mom_dec', 'risevsdec_mom'])

        mat = scipy.io.loadmat(parentFolderPath + 'Cell' + str(cellNum) + '_1s/TagRFP/Tracking/ProcessedTracks.mat')
        tracks = mat['tracks'][0, :]

        for i in range(len(tracks)):
            # TrackNum:
            trackNum = str(cellNum) + "-" + str(i + 1)
            if i == 0:
                print('Querring data for cell number {} of 8, number of tracks={}. '.format(cellNum,  len(tracks)-1), end="")

            # Intensity (A)
            intensity = tracks[i][4]
            intensity_t1 = list(intensity[0])
            intensity_t2 = list(intensity[1])
            intensity_max = max(intensity_t1)
            lifetime = len(intensity_t1)
            max_index = intensity_t1.index(intensity_max)

            # Background(c)
            bkground = tracks[i][5]
            bkground_t1 = list(bkground[0])
            bkground_t2 = list(bkground[1])
            bkground_scal = bkground_t1[max_index]

            # Category(catldx)
            catldx = list(chain.from_iterable(tracks[i][33]))
            catldx_scal = catldx[0]

            # Pvals(pvals)
            pvals = tracks[i][12]
            pvals_t2 = list(pvals[1])

            # Frame(f)
            frame = list(chain.from_iterable(tracks[i][1]))
            frame_scal = frame[max_index]

            # Start Buffer(startBuffer)
            startBuffer = list(chain.from_iterable(tracks[i][24]))

            # End Buffer(endBuffer)
            endBuffer = list(chain.from_iterable(tracks[i][25]))

            # Motion Analysis:
            motion_analysis = list(chain.from_iterable(tracks[i][26]))
            if not motion_analysis:  # If motion analysis is empty...
                tdisp_scal = None
                msd_max = None
            else:
                tdisp = list(chain.from_iterable(motion_analysis[0][0]))
                tdisp_scal = tdisp[0]
                msd = list(chain.from_iterable(motion_analysis[0][1]))
                msd_max = max(msd)

            # Check if Aux+ or Aux-
            # Add classification parameters to this section
            isAux = False
            pval_cutoff = pvalCutOff
            # Look for Consecutive True's
            if (lifetime < 20 and checkLifetime):
                numConsecPValTemp = numConsecPVal - 1
            else:
                numConsecPValTemp = numConsecPVal
            for x in range(len(pvals_t2) - (numConsecPValTemp - 1)):
                isSig = False
                for j in range(x, x + (numConsecPValTemp)):
                    isSig = pvals_t2[j] <= pval_cutoff
                    if not isSig:
                        break
                if mustBeSecondHalf:
                    if (x < (int(lifetime) / 2)):
                        isSig = False
                if isSig:
                    isAux = True
            if isAux:
                # Classify as Aux +
                aux = 1
            else:
                # Classify as Aux -
                aux = 0

            # Generate Full Intensity for Min / Max Rise / Decay
            if startBuffer and endBuffer:
                intensity_list = list(startBuffer[0][3][0]) + intensity_t1 + list(endBuffer[0][3][0])
                start_index = 1
                end_index = len(intensity_list) - 1
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

                # Get Moments
                rise_slope_mom = list()
                decay_slope_mom = list()
                start_index = 1
                end_index = len(total_slope) - 1
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
                # endif
            # endif

            # Create Row
            row = [trackNum, frame_scal, lifetime, intensity_max, bkground_scal, tdisp_scal, msd_max, catldx_scal, aux,
                   avgRise, avgDecay, riseVsDecay, avgRiseMom, avgDecayMom, riseVsDecayMom]

            # Add to DF
            df_temp.loc[i] = row


        df = pd.concat([df , df_temp], ignore_index=False)
        print("Completed")
    df = df.set_index('trackNum')
    df = filterMatrixData(data = df, dropNA = True, catsToUse = [1], colsToDrop = ['frame'])
    if not os.path.exists(parentFolderPath + os.path.sep + 'dataframe'):
        os.makedirs(parentFolderPath + os.path.sep + 'dataframe')
        df.to_csv(parentFolderPath + os.path.sep + 'dataframe' + os.path.sep + "aux.csv")
        print("dataframe saved to {}".format(parentFolderPath + os.path.sep + 'dataframe' + os.path.sep + "aux.csv"))

    return df

def filterMatrixData(data, dropNA = False, catsToUse = [1,2,3,4,5,6,7,8], colsToDrop = []):
    if dropNA:
        data = data.dropna()
    data = data.loc[data['catIdx'].isin(catsToUse)]
    for col in colsToDrop:
        data = data.drop(labels = col, axis = 1)
    return(data)


def visualize():

    pass

def normalize_df(df):
    print('Normalizing the dataframe')
    df_norm = df.copy()
    print(df_norm)
    df_scaled = pd.DataFrame(StandardScaler().fit_transform(df_norm), columns=df_norm.columns)
    df_norm[:] = df_scaled[:].values

    # NOTE: Change below to change them individually -------
    df_norm['aux'] = df['aux'].values
    df_norm['lifetime'] = df['lifetime'].values
    numbins = 8

    zero_data = np.zeros(shape=(df.shape[0], numbins))
    df_zero = pd.DataFrame(zero_data, columns=['cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8'],
                           index=df.index)
    j = 0
    for i in df.index.values:
        val = df['catIdx'][i]
        df_zero.iat[j, int(val) - 1] = 1
        j = j + 1

    continuousLabels = ['aux', 'lifetime', 'max_intensity', 'background', 'totaldisp', 'max_msd', 'avg_rise', 'avg_dec', 'risevsdec', 'avg_mom_rise', 'avg_mom_dec', 'risevsdec_mom']
    dropCols = ['frame']
    continuousLabels = [i for i in continuousLabels if i not in dropCols]

    # Combine Continuous w/ Categorical:
    output_df = pd.concat([df_norm[continuousLabels], df_zero], axis=1, ignore_index=False)
    df_total = output_df

    #Separate (as a checkpoint) the aux + and aux - to describe them
    df_aux0 = df_total.loc[df_total['aux'].isin([0])]
    df_aux1 = df_total.loc[df_total['aux'].isin([1])]

    print("Aux + and - info:")
    print(df_aux0.describe())
    print(df_aux1.describe())

    # print("RUN " + str(i) + ": Cats Used: " + str(catsToUse[i]) + " | pval: " + str(pvalCutOff[i]) + " | #ofPVal: " + str(numConsecPVal[i]) + " | Half? " + str(mustBeSecondHalf[i]) + " | Apply LT? " + str(checkLifetime[i]))
    # print("# Qualifying Tracks: " + str(len(df_total.values)) + " | # Aux+: " + str(len(df_aux1.values))  + " | # Aux-: " + str(len(df_aux0.values)))
    # print(df_aux1)
    # print()
    return (output_df)


def preprocess_features(df):
    print("\n2\n")
    """Prepares input features from the data set.

    Args:
        df: A Pandas DataFrame expected to contain data
        from the data set.
    Returns:
        A DataFrame that contains the features to be used for the model, including
        synthetic features.
    """
    selected_features = df[
        ['lifetime', 'max_intensity', 'background',
        'totaldisp', 'max_msd',
        'avg_rise', 'avg_dec', 'risevsdec',
        'avg_mom_rise', 'avg_mom_dec', 'risevsdec_mom',
        'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8']]
    processed_features = selected_features.copy()
    return processed_features


def construct_feature_columns(input_features):
    print("\n4\n")
    # Construct the TensorFlow Feature Columns.
    #
    # Args:
    #    input_features: The names of the numerical input features to use.
    # Returns:
    #    A set of feature columns

    return set([tf.feature_column.numeric_column(my_feature)
                for my_feature in input_features])


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    print("\n5\n")
    """Trains a linear regression model.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def preprocess_targets(df):
    print("\n3\n")
    """Prepares target features (i.e., labels) from the data set.

    Args:
        df: A Pandas DataFrame expected to contain data
        from the data set.
    Returns:
        A DataFrame that contains the target feature.
    """
    output_targets = pd.DataFrame()
    # Scale the target to be in units of thousands of dollars.
    output_targets["aux"] = df["aux"]
    return output_targets

def train_model(  # happens late
        learning_rate,
        steps,
        batch_size,
        df_total,
        percentTrain,
        percentTest,
        percentTrainToVal,
        targetCol='aux',
        showTrainingPrints=True,
        shuffleTrainingData=True,
        numPeriods=20):
    print("\n6\n")
    df_total = df_total.astype(float)
    numTracks = df_total.shape[0]
    # Trains a linear regression model.
    #
    # In addition to training, this function also prints training progress information,
    # as well as a plot of the training and validation loss over time.
    #
    # Args:
    #    learning_rate: A `float`, the learning rate.
    #    steps: A non-zero `int`, the total number of training steps. A training step
    #    consists of a forward and backward pass using a single batch.
    #    feature_columns: A `set` specifying the input feature columns to use.
    #    training_examples: A `DataFrame` containing one or more columns from
    #    `df` to use as input features for training.
    #    training_targets: A `DataFrame` containing exactly one column from
    #    `df` to use as target for training.
    #    validation_examples: A `DataFrame` containing one or more columns from
    #    `df` to use as input features for validation.
    #    validation_targets: A `DataFrame` containing exactly one column from
    #    `df` to use as target for validation.
    #
    # Returns:
    #    A `DNNClassifier` object trained on the training data.
    #    3 CSV Files labeled 'total_pred', 'pred_correct' and 'pred_incorrect'

    # Re-Index Randomly:
    df_sample = df_total.sample(frac=1)

    # Perform initial split into Training and Testing:
    numTotal = df_sample.shape[0]
    numTotalTrain = math.ceil(percentTrain * numTotal)
    numTotalTest = numTotal - numTotalTrain
    df_train = df_sample.head(numTotalTrain)
    df_test = df_sample.tail(numTotalTest)

    # Choose training.
    numTrain = math.ceil(percentTrainToVal * numTotalTrain)
    training_examples = preprocess_features(df_train.head(numTrain))
    training_targets = preprocess_targets(df_train.head(numTrain))

    # Choose validation.
    numValidation = numTotalTrain - numTrain
    validation_examples = preprocess_features(df_train.tail(numValidation))
    validation_targets = preprocess_targets(df_train.tail(numValidation))

    # Choose testing.
    test_examples = preprocess_features(df_test)
    test_targets = preprocess_targets(df_test)

    # Double-check that we've done the right thing.
    print("Training examples summary:")
    display.display(training_examples.describe())
    print(training_examples)
    print("Validation examples summary:")
    display.display(validation_examples.describe())
    print(validation_examples)
    print("Testing examples summary:")
    display.display(test_examples.describe())
    print(test_examples)

    print("Training targets summary:")
    display.display(training_targets.describe())
    print(training_targets)
    print("Validation targets summary:")
    display.display(validation_targets.describe())
    print(validation_targets)
    print("Testing targets summary:")
    display.display(test_targets.describe())
    print(test_targets)

    # Get Feature Columns
    feature_columns = construct_feature_columns(training_examples)

    periods = numPeriods
    steps_per_period = steps / periods

    dnn_classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 10],
        n_classes=2
    )

    training_input_fn = lambda: my_input_fn(training_examples,
                                            training_targets[targetCol],
                                            batch_size=batch_size)
    eval_input_fn = lambda: my_input_fn(validation_examples,
                                        validation_targets[targetCol],
                                        num_epochs=1,
                                        shuffle=False)
    test_input_fn = lambda: my_input_fn(test_examples,
                                        test_targets[targetCol],
                                        num_epochs=1,
                                        shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("Accuracy: (on Validation data):")
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        dnn_classifier.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # Evaluate Validation Set:
        eval_result = dnn_classifier.evaluate(
            input_fn=eval_input_fn)

        print('\nValidation set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

        if shuffleTrainingData:
            # Shuffle the Training and Validation Data for the next run
            # df_train has ALL of the data for the training and validation sets
            df_train_sample = df_train.sample(frac=1)

            # Choose new training.
            numTrain = math.ceil(percentTrainToVal * numTotalTrain)
            training_examples = preprocess_features(df_train_sample.head(numTrain))
            training_targets = preprocess_targets(df_train_sample.head(numTrain))

            # Choose new validation.
            numValidation = numTotalTrain - numTrain
            validation_examples = preprocess_features(df_train_sample.tail(numValidation))
            validation_targets = preprocess_targets(df_train_sample.tail(numValidation))

        if showTrainingPrints:
            print("Training examples summary:")
            display.display(training_examples.describe())
            print(training_examples)
            print("Validation examples summary:")
            display.display(validation_examples.describe())
            print(validation_examples)

            print("Training targets summary:")
            display.display(training_targets.describe())
            print(training_targets)
            print("Validation targets summary:")
            display.display(validation_targets.describe())
            print(validation_targets)

    print("Model training finished.")

    # Evaluate Test Set:
    test_result = dnn_classifier.evaluate(
        input_fn=test_input_fn)

    # 0 - 'Aux -'
    # 1 - 'Aux +'
    expected = [0, 1]
    predictions = dnn_classifier.predict(
        input_fn=test_input_fn)
    pred_list = list()
    for i in predictions:
        pred_list.append(i['class_ids'][0])

    cell_indexes = test_targets.index.to_series()
    cell_nums = [x.split("-")[0] for x in cell_indexes]
    track_nums = [x.split("-")[1] for x in cell_indexes]
    cell_num_df = pd.DataFrame(cell_nums, index=test_targets.index)
    track_num_df = pd.DataFrame(track_nums, index=test_targets.index)
    pred_df = pd.DataFrame(pred_list, index=test_targets.index)

    total_pred = pd.concat([cell_num_df, track_num_df, pred_df, test_targets], axis=1, ignore_index=False)
    total_pred.columns = ['cell', 'track', 'predictions', 'values']

    pred_correct = total_pred.loc[total_pred['predictions'] == total_pred['values']]
    pred_incorrect = total_pred.loc[total_pred['predictions'] != total_pred['values']]

    pred_dir = '/Users/g/PycharmProjects/pythonProject2/python/predictions/'
    print("Predictions:")
    total_pred.to_csv(pred_dir + '/total_pred.csv', sep=',', encoding='utf-8')
    print(total_pred)

    print("Correct:")
    pred_correct.to_csv(pred_dir + '/pred_correct.csv', sep=',', encoding='utf-8')
    print(pred_correct)

    print("Incorrect:")
    pred_incorrect.to_csv(pred_dir + '/pred_incorrect.csv', sep=',', encoding='utf-8')
    print(pred_incorrect)

    print(' --- FINAL Test set accuracy: {accuracy:0.3f}\n'.format(**test_result))

    return dnn_classifier


if __name__ == "__main__":

    # organize input arguments
    verbose, inputData_dir = get_args(sys.argv[1:])
    df = get_inputData(inputData_dir, verbose)
    df = normalize_df(df)
    model = train_model(
        learning_rate=.0001,
        steps=1000,
        batch_size=100,
        df_total=df,
        percentTrain=0.8,
        percentTest=1-0.8,
        percentTrainToVal=0.8,
        shuffleTrainingData=True,
        showTrainingPrints=False,
        numPeriods=20,
    )

    # print(tf.__version__)

