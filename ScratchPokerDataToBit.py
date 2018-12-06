import numpy as np


class ScratchPokerDataToBit(object):
    # most ML data is structured with the output as the last column, so this splits that off into y
    def __init__(self, fileName, dataType):
        suit1 = [1, 0, 0, 0]
        suit2 = [0, 1, 0, 0]
        suit3 = [0, 0, 1, 0]
        suit4 = [0, 0, 0, 1]

        card1 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        card2 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        card3 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        card4 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        card5 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        card6 = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        card7 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        card8 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        card9 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        card10 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        card11 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        card12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        card13 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

        hand0 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        hand1 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        hand2 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        hand3 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        hand4 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        hand5 = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        hand6 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        hand7 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        hand8 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        hand9 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

        data = np.loadtxt(fileName, dtype=dataType, delimiter=',')
        self.X = data[0:, 0:-1]
        xTemp = list()
        for x in range(len(self.X)):
            rowTemp = list()
            for y in range(int(len(self.X[x]) / 2)):
                suitIndexTemp = int(y * 2)
                cardIndexTemp = int(y * 2 + 1)
                if self.X[x][suitIndexTemp] == 1:
                    rowTemp[len(rowTemp):len(rowTemp)] = suit1
                elif self.X[x][suitIndexTemp] == 2:
                    rowTemp[len(rowTemp):len(rowTemp)] = suit2
                elif self.X[x][suitIndexTemp] == 3:
                    rowTemp[len(rowTemp):len(rowTemp)] = suit3
                elif self.X[x][suitIndexTemp] == 4:
                    rowTemp[len(rowTemp):len(rowTemp)] = suit4
                if self.X[x][cardIndexTemp] == 1:
                    rowTemp[len(rowTemp):len(rowTemp)] = card1
                elif self.X[x][cardIndexTemp] == 2:
                    rowTemp[len(rowTemp):len(rowTemp)] = card2
                elif self.X[x][cardIndexTemp] == 3:
                    rowTemp[len(rowTemp):len(rowTemp)] = card3
                elif self.X[x][cardIndexTemp] == 4:
                    rowTemp[len(rowTemp):len(rowTemp)] = card4
                elif self.X[x][cardIndexTemp] == 5:
                    rowTemp[len(rowTemp):len(rowTemp)] = card5
                elif self.X[x][cardIndexTemp] == 6:
                    rowTemp[len(rowTemp):len(rowTemp)] = card6
                elif self.X[x][cardIndexTemp] == 7:
                    rowTemp[len(rowTemp):len(rowTemp)] = card7
                elif self.X[x][cardIndexTemp] == 8:
                    rowTemp[len(rowTemp):len(rowTemp)] = card8
                elif self.X[x][cardIndexTemp] == 9:
                    rowTemp[len(rowTemp):len(rowTemp)] = card9
                elif self.X[x][cardIndexTemp] == 10:
                    rowTemp[len(rowTemp):len(rowTemp)] = card10
                elif self.X[x][cardIndexTemp] == 11:
                    rowTemp[len(rowTemp):len(rowTemp)] = card11
                elif self.X[x][cardIndexTemp] == 12:
                    rowTemp[len(rowTemp):len(rowTemp)] = card12
                elif self.X[x][cardIndexTemp] == 13:
                    rowTemp[len(rowTemp):len(rowTemp)] = card13
                # print('x: ' + str(x) + ' rowTemp: ' + str(rowTemp))
            xTemp.append(rowTemp)
        xTemp = np.array(xTemp)
        self.X = xTemp
        self.y = data[0:, -1]
        self.y = self.y.reshape(self.y.shape[0], 1)
        yTemp = self.y.tolist()
        for x in range(len(yTemp)):
            if yTemp[x][0] == 0:
                yTemp[x] = hand0
            elif yTemp[x][0] == 1:
                yTemp[x] = hand1
            elif yTemp[x][0] == 2:
                yTemp[x] = hand2
            elif yTemp[x][0] == 3:
                yTemp[x] = hand3
            elif yTemp[x][0] == 4:
                yTemp[x] = hand4
            elif yTemp[x][0] == 5:
                yTemp[x] = hand5
            elif yTemp[x][0] == 6:
                yTemp[x] = hand6
            elif yTemp[x][0] == 7:
                yTemp[x] = hand7
            elif yTemp[x][0] == 8:
                yTemp[x] = hand8
            elif yTemp[x][0] == 9:
                yTemp[x] = hand9
        yTemp = np.array(yTemp)
        self.y = yTemp

    def getXandy(self):
        return self.X, self.y

