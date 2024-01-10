"""

@author: Clay Preusch & Grant Konkel

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
import math
import pandas as pd


def squaredError(sampleA, sampleB):
    sum = 0

    # lat and long error
    # euclidean distance scaled up bc lat and long should be highly dependent on the decimals
    sum += (((sampleA[0]-sampleB[0])*1000) ** 2 +
            (((sampleA[1]-sampleB[1]))*1000) ** 2)

    # speed error
    sum += ((sampleA[2]-sampleB[2])/10) ** 2

    # theta error (dependent on mean speed between two points)
    meanSpeed = (sampleA[2] + sampleB[2])/2
    theta = sampleA[3] - sampleB[3]

    # find min theta between point A and B
    if theta > 1800:
        theta = abs(theta-3600)

    sum += (theta*meanSpeed/10000) ** 2

    return sum


def magic(boats, init_assignments, k):

    # remove empty boats
    boats = [boat for boat in boats if boat]

    se = [["N/A", "N/A", float("inf")] for boat in boats]

    print(len(se))
    i = 0

    for i in range(len(boats)):
        for j in range(i+1, len(boats)):
            # if boats[i] != boats[j]:
            if abs(boats[i][0][0] - boats[j][0][0]) < pd.Timedelta('1 minute'):
                sampleBoat = [boats[i][0]['LAT'], boats[i][0]['LON'], boats[i]
                              [0]['SPEED_OVER_GROUND'], boats[i][0]['COURSE_OVER_GROUND']]
                movingBoat = boats[j][0]

                t_dif = boats[i][0][0] - boats[j][0][0]

                dist = movingBoat[3] / 10 * t_dif.total_seconds() / 3600

                ex_lat = movingBoat[1] + dist * \
                    math.cos(math.radians(movingBoat[4] / 10))
                ex_long = movingBoat[2] + dist * \
                    math.sin(math.radians(movingBoat[4] / 10))

                expectedBoat = [ex_lat, ex_long, movingBoat[3], movingBoat[4]]
                se_temp = squaredError(expectedBoat, sampleBoat)

                if se[i][2] > se_temp:
                    se[i][2] = se_temp
                    se[i][1] = j
                    se[i][0] = i

            if abs(boats[i][-1][0] - boats[j][0][0]) < pd.Timedelta('1 minute'):
                sampleBoat = [boats[i][-1]['LAT'], boats[i][-1]['LON'], boats[i]
                              [-1]['SPEED_OVER_GROUND'], boats[i][-1]['COURSE_OVER_GROUND']]
                movingBoat = boats[j][0]

                t_dif = boats[i][-1][0] - boats[j][0][0]

                dist = movingBoat[3] / 10 * t_dif.total_seconds() / 3600

                ex_lat = movingBoat[1] + dist * \
                    math.cos(math.radians(movingBoat[4] / 10))
                ex_long = movingBoat[2] + dist * \
                    math.sin(math.radians(movingBoat[4] / 10))

                expectedBoat = [ex_lat, ex_long, movingBoat[3], movingBoat[4]]
                se_temp = squaredError(expectedBoat, sampleBoat)

                if se[i][2] > se_temp:
                    se[i][2] = se_temp
                    se[i][1] = j
                    se[i][0] = i

            if abs(boats[i][0][0] - boats[j][-1][0]) < pd.Timedelta('1 minute'):
                sampleBoat = [boats[i][0]['LAT'], boats[i][0]['LON'], boats[i]
                              [0]['SPEED_OVER_GROUND'], boats[i][0]['COURSE_OVER_GROUND']]
                movingBoat = boats[j][-1]

                t_dif = boats[i][0][0] - boats[j][-1][0]

                dist = movingBoat[3] / 10 * t_dif.total_seconds() / 3600

                ex_lat = movingBoat[1] + dist * \
                    math.cos(math.radians(movingBoat[4] / 10))
                ex_long = movingBoat[2] + dist * \
                    math.sin(math.radians(movingBoat[4] / 10))

                expectedBoat = [ex_lat, ex_long, movingBoat[3], movingBoat[4]]
                se_temp = squaredError(expectedBoat, sampleBoat)

                if se[i][2] > se_temp:
                    se[i][2] = se_temp
                    se[i][1] = j
                    se[i][0] = i

            if abs(boats[i][-1][0] - boats[j][-1][0]) < pd.Timedelta('1 minute'):
                sampleBoat = [boats[i][-1]['LAT'], boats[i][-1]['LON'], boats[i]
                              [-1]['SPEED_OVER_GROUND'], boats[i][-1]['COURSE_OVER_GROUND']]
                movingBoat = boats[j][-1]

                t_dif = boats[i][-1][0] - boats[j][-1][0]

                dist = movingBoat[3] / 10 * t_dif.total_seconds() / 3600

                ex_lat = movingBoat[1] + dist * \
                    math.cos(math.radians(movingBoat[4] / 10))
                ex_long = movingBoat[2] + dist * \
                    math.sin(math.radians(movingBoat[4] / 10))

                expectedBoat = [ex_lat, ex_long, movingBoat[3], movingBoat[4]]
                se_temp = squaredError(expectedBoat, sampleBoat)

                if se[i][2] > se_temp:
                    se[i][2] = se_temp
                    se[i][1] = j
                    se[i][0] = i

    se_sorted = sorted(se, key=lambda x: x[2])
    se_sorted = se_sorted[:len(boats)-k]

    se_sorted = sorted(se_sorted, key=lambda x: x[0])

    for small_se_pair in se_sorted:

        # AHHHHHHHHHHHHHHHHHHHHH
        # update assignments
        for i in range(len(init_assignments)):
            if init_assignments[i] == small_se_pair[0]:
                init_assignments[i] = small_se_pair[1]

    return init_assignments


def predictWithK(testFeatures, numVessels, trainFeatures=None, trainLabels=None):

    boats = [[] for _ in range(10000)]
    boats_ID = [[] for _ in range(10000)]

    testFeatures[:, 0] = pd.to_datetime(testFeatures[:, 0])

    X_copy = pd.DataFrame(testFeatures, columns=[
                          'SEQUENCE_DTTM', 'LAT', 'LON', 'SPEED_OVER_GROUND', 'COURSE_OVER_GROUND'])
    X_copy['SEQUENCE_DTTM'] = pd.to_datetime(X_copy['SEQUENCE_DTTM'])

    i = 0

    while len(X_copy) > 0:
        if len(X_copy) == 0:
            break

        boats[i].append(X_copy.iloc[0])
        boats_ID[i].append(X_copy.index[0])
        X_copy = X_copy.drop(X_copy.index[0])

        for sample_idx, sample in X_copy.iterrows():
            t_dif = sample['SEQUENCE_DTTM'] - boats[i][-1]['SEQUENCE_DTTM']

            if t_dif > pd.Timedelta('1 minutes'):
                break

            dist = boats[i][-1]['SPEED_OVER_GROUND'] / \
                10 * t_dif.total_seconds() / 3600

            ex_lat = boats[i][-1]['LAT'] + dist * \
                math.cos(math.radians(boats[i][-1]['COURSE_OVER_GROUND'] / 10))
            ex_long = boats[i][-1]['LON'] + dist * \
                math.sin(math.radians(boats[i][-1]['COURSE_OVER_GROUND'] / 10))

            expectedBoat = [ex_lat, ex_long, boats[i][-1]
                            ['SPEED_OVER_GROUND'], boats[i][-1]['COURSE_OVER_GROUND']]
            sampleBoat = [sample['LAT'], sample['LON'],
                          sample['SPEED_OVER_GROUND'], sample['COURSE_OVER_GROUND']]
            se = squaredError(expectedBoat, sampleBoat)
            if se < 1:
                boats[i].append(sample)
                boats_ID[i].append(sample_idx)
                X_copy = X_copy.drop(sample_idx)

        i += 1

    classified_points = []
    for classification, boat_ids in enumerate(boats_ID):
        classified_points.extend([classification] * len(boat_ids))

    # Sort classified_points by boat IDs
    ordered_points = [x for _, x in sorted(
        zip(sum(boats_ID, []), classified_points))]

    assignments = magic(boats, ordered_points, numVessels)

    # print(assignments)

    return np.array(assignments)


def predictWithoutK(testFeatures, trainFeatures=None, trainLabels=None):

    boats = [[] for _ in range(10000)]
    boats_ID = [[] for _ in range(10000)]

    testFeatures[:, 0] = pd.to_datetime(testFeatures[:, 0])

    X_copy = pd.DataFrame(testFeatures, columns=[
                          'SEQUENCE_DTTM', 'LAT', 'LON', 'SPEED_OVER_GROUND', 'COURSE_OVER_GROUND'])
    X_copy['SEQUENCE_DTTM'] = pd.to_datetime(X_copy['SEQUENCE_DTTM'])

    i = 0

    while len(X_copy) > 0:
        if len(X_copy) == 0:
            break

        boats[i].append(X_copy.iloc[0])
        boats_ID[i].append(X_copy.index[0])
        X_copy = X_copy.drop(X_copy.index[0])

        for sample_idx, sample in X_copy.iterrows():
            t_dif = sample['SEQUENCE_DTTM'] - boats[i][-1]['SEQUENCE_DTTM']

            if t_dif > pd.Timedelta('1 minutes'):
                break

            dist = boats[i][-1]['SPEED_OVER_GROUND'] / \
                10 * t_dif.total_seconds() / 3600

            ex_lat = boats[i][-1]['LAT'] + dist * \
                math.cos(math.radians(boats[i][-1]['COURSE_OVER_GROUND'] / 10))
            ex_long = boats[i][-1]['LON'] + dist * \
                math.sin(math.radians(boats[i][-1]['COURSE_OVER_GROUND'] / 10))

            expectedBoat = [ex_lat, ex_long, boats[i][-1]
                            ['SPEED_OVER_GROUND'], boats[i][-1]['COURSE_OVER_GROUND']]
            sampleBoat = [sample['LAT'], sample['LON'],
                          sample['SPEED_OVER_GROUND'], sample['COURSE_OVER_GROUND']]
            se = squaredError(expectedBoat, sampleBoat)
            if se < 1:
                boats[i].append(sample)
                boats_ID[i].append(sample_idx)
                X_copy = X_copy.drop(sample_idx)

        i += 1

    classified_points = []
    for classification, boat_ids in enumerate(boats_ID):
        classified_points.extend([classification] * len(boat_ids))

    # Sort classified_points by boat IDs
    ordered_points = [x for _, x in sorted(
        zip(sum(boats_ID, []), classified_points))]

    assignments = magic(boats, ordered_points, 21)

    return np.array(assignments)


# ONLY RUNS IF RUN AS MAIN SCRIPT
if __name__ == "__main__":

    from utils import loadData, plotVesselTracks

    data = loadData('set2.csv')

    features = data[:, 2:]
    labels = data[:, 1]

    # %% plot all vessel tracks with no coloring

    plotVesselTracks(features[:, [2, 1]])
    plt.title('All vessel tracks')

    # %% run prediction algorithms and check accuracy

    # prediction with specified number of vessels
    numVessels = np.unique(labels).size
    predVesselsWithK = predictWithK(features, numVessels)
    ariWithK = adjusted_rand_score(labels, predVesselsWithK)

    # prediction without specified number of vessels
    predVesselsWithoutK = predictWithoutK(features)
    predNumVessels = np.unique(predVesselsWithoutK).size
    ariWithoutK = adjusted_rand_score(labels, predVesselsWithoutK)

    print(f'Adjusted Rand index given K = {numVessels}: {ariWithK}')
    print(f'Adjusted Rand index for estimated K = {predNumVessels}: '
          + f'{ariWithoutK}')

    # %% plot vessel tracks colored by prediction and actual labels
    plotVesselTracks(features[:, [2, 1]], predVesselsWithK)
    plt.title('Vessel tracks by cluster with K')
    plotVesselTracks(features[:, [2, 1]], predVesselsWithoutK)
    plt.title('Vessel tracks by cluster without K')
    plotVesselTracks(features[:, [2, 1]], labels)
    plt.title('Vessel tracks by label')
