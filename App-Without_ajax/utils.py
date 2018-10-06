import random
import string


def random_string():
    rs = (''.join(random.choice(string.ascii_uppercase)
                  for i in range(16)))
    return rs


def jaccard_score(string1, string2):
    a = []
    b = []
    for i in string1:
        a.append(i)
    for i in string2:
        b.append(i)
    num = len(set(a).intersection(set(b)))
    print(num)
    deno = (len(a)) + len(b) - num
    print(deno)
    return (num / deno)


if __name__ == '__main__':
    a = "aab"
    v = "abb"
    c = "aabaaaaaaaaaaaa"
    print(jaccard_score(a, c))
