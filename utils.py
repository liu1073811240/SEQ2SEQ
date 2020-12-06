import random


# a = random.randint(65, 66)
# print(a)

nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
lower_char = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
              'v', 'w', 'x', 'y', 'z']
upper_char = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
              'V', 'W', 'X', 'Y', 'Z']


def StrtoLabel(Str):

    label = []
    for i in range(0, 4):
        if Str[i] >= '0' and Str[i] <= '9':
            # print(ord(Str[i]))  # 1通过ascll转为49
            # print(ord('0'))  # 0通过ascll转为48
            label.append(ord(Str[i]) - ord('0'))
        elif Str[i] >= 'a' and Str[i] <= 'z':

            label.append(ord(Str[i]) - ord('a') + 10)
        else:
            label.append(ord(Str[i]) - ord('A') + 36)
    return label


def LabeltoStr(Label):
    Str = ""
    for i in Label:
        if i <= 9:
            Str += chr(ord('0') + i)
        elif i <= 35:
            Str += chr(ord('a') + i - 10)

        else:
            Str += chr(ord('A') + i - 36)
    return Str


if __name__ == '__main__':

    a = StrtoLabel("1aAB")
    print(a)  # [1, 10, 36, 37]

    b = LabeltoStr(a)
    print(b)  # 1aAB

