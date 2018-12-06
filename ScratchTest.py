from ScratchPokerDataToBit import *


suitDict = {'[1000]': 'Clubs', '[0100]': 'Diamonds', '[0010]': 'Hearts', '[0001]': 'Spades'}
cardTypeDict = {'[1000000000000]': 'Ace',   '[0100000000000]': 'Two',
                '[0010000000000]': 'Three', '[0001000000000]': 'Four',
                '[0000100000000]': 'Five',  '[0000010000000]': 'Six',
                '[0000001000000]': 'Seven', '[0000000100000]': 'Eight',
                '[0000000010000]': 'Nine',  '[0000000001000]': 'Ten',
                '[0000000000100]': 'Jack',  '[0000000000010]': 'Queen',
                '[0000000000001]': 'King'}
handDict = {'[1000000000]': 'Nothing in hand', '[0100000000]': 'One pair',
            '[0010000000]': 'Two pairs',       '[0001000000]': 'Three of a kind',
            '[0000100000]': 'Straight',        '[0000010000]': 'Flush',
            '[0000001000]': 'Full house',      '[0000000100]': 'Four of a kind',
            '[0000000010]': 'Straight flush',  '[0000000001]': 'Royal flush'}

def handBinaryToWords(input):
    hand = ''
    for x in range(5):
        tempSuitBinaryString = str(input[0:4]).replace(' ', '')
        tempCardBinaryString = str(input[4:17]).replace(' ', '')
        input = input[17:len(input)]
        hand = hand + cardTypeDict.get(tempCardBinaryString) + ' of ' + suitDict.get(tempSuitBinaryString) + ', '

    return hand[0:len(hand) - 2]

def handTypeBinaryToWords(input):
    return handDict.get(str(input).replace(' ', ''))


s = ScratchPokerDataToBit('pokerData.txt', 'int')
X, y = s.getXandy()

index = 100
hand1 = X[index]
handType1 = y[index]
print(handBinaryToWords(hand1))
print(handTypeBinaryToWords(handType1))

