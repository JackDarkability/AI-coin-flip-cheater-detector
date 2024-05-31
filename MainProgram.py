import csv
import random
import numpy as np
from sklearn.svm import SVC


def main():
    results = coinThrowerAlgorithm(samples=10000, maxNumThrows=20)
    testResults(results, testCheater=True)
    testResults(results, testCheater=False)
    fieldNames = ["numHeads", "numTails", "cheater"]
    writeResults(fieldNames, results)
    model = trainModel()
    gameLoop(model)


def gameLoop(model, needMoreDataProbability=0.8):
    """
    The final game loop. The user will enter how many heads and tails have been flipped
    The model will then output its classification and percentage likelihood of being correct
    """

    while True:
        heads = int(input("How many heads?: "))
        tails = int(input("How many tails?: "))
        resultProbs = model.predict_proba(np.array([heads, tails]).reshape(1, -1))
        if np.max(resultProbs) < needMoreDataProbability:
            print("Need more data!")
        result = model.predict(np.array([heads, tails]).reshape(1, -1))
        output = "I am " + str(np.max(resultProbs) * 100) + "% sure they are "

        if str(result[0]) == "False":  # If not a cheater
            output = output + "not "

        output = output + "a cheater"
        print(output)


def writeResults(fieldNames, results, filename="records.csv"):
    """
    Write results to csv file
    """
    f = open(filename, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldNames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)
    f.close()


def coinThrowerAlgorithm(
    samples=1000,
    percentageHeadOfCheater=0.75,
    minNumThrows=5,
    maxNumThrows=10,
):
    """
    Runs coin throwing process for amount of samples and writes to file in format numHeads,numTails,cheater
    Percentage head of cheater is value between 0 and 1 of how often head comes up for cheaters
    """

    results = []  # Array of dictionaries in format NumHeads,NumTails,isCheater
    for sample in range(samples):
        iterations = random.randint(minNumThrows, maxNumThrows)
        numOfHeads = 0
        numOfTails = 0
        isCheater = random.choice([True, False])

        if isCheater == True:
            probabilityOfHeads = percentageHeadOfCheater
        else:
            probabilityOfHeads = 0.5

        for iteration in range(iterations):
            throw = random.random()  # Between 0 and 1
            if throw < probabilityOfHeads:  # So if throw is heads
                numOfHeads += 1
            else:
                numOfTails += 1

        result = {
            "numHeads": numOfHeads,
            "numTails": numOfTails,
            "cheater": str(isCheater),
        }

        results.append(result)

    return results


def testResults(results, testCheater=True):
    """
    Get testing data for probability of heads in data for specified group
    If testCheater == false then test fair people
    """
    numPeople = 0
    totalNumHeads = 0
    totalNumTails = 0

    for result in results:
        numHeads = result["numHeads"]
        numTails = result["numTails"]
        isCheater = result["cheater"]
        if isCheater == str(testCheater):
            totalNumHeads += numHeads
            totalNumTails += numTails
            numPeople += 1

    if testCheater == True:
        peopleTesting = "cheaters"

    else:
        peopleTesting = "fair people"

    print(
        "Total number of "
        + peopleTesting
        + " was "
        + str(numPeople)
        + " and there was a "
        + str(totalNumHeads / (totalNumHeads + totalNumTails))
        + "% head rate"
    )


def trainModel(cValue=1.0, numFeatures=2, fileName="records.csv"):
    """
    Train the SVC model based on the training data collected
    """
    arr = np.genfromtxt(fileName, delimiter=",", dtype=str)
    dataSampleResults = arr[1:]  # Ignore header

    x = dataSampleResults[:, 0:numFeatures]
    # Get first numFeatures columns as classification data

    y = dataSampleResults[:, numFeatures]  # Get last column as target
    svcClassifier = SVC(C=cValue, probability=True)
    svcClassifier.fit(x, y)
    return svcClassifier


if __name__ == "__main__":
    main()
