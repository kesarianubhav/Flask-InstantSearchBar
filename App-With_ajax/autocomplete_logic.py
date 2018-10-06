import numpy as np
import pandas as pd
import time
import os


class TrieNode:

    def __init__(self):
        self.end = False
        self.children = {}

    def all_words(self, prefix):
        if self.end:
            yield prefix

        for letter, child in self.children.items():
            yield from child.all_words(prefix + letter)


class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        curr = self.root
        for letter in word:
            node = curr.children.get(letter)
            if not node:
                node = TrieNode()
                curr.children[letter] = node
            curr = node
        curr.end = True

    def search(self, word):
        curr = self.root
        for letter in word:
            node = curr.children.get(letter)
            if not node:
                return False
            curr = node
        return curr.end

    def all_words_beginning_with_prefix(self, prefix):
        cur = self.root
        for c in prefix:
            cur = cur.children.get(c)
            if cur is None:
                return  # No words with given prefix

        yield from cur.all_words(prefix)


def get_best_matches(trie, string, no=10):
    results = list(trie.all_words_beginning_with_prefix(string))
    return results[:no]


def loadData(filename):
    # with open('')
    df = pd.read_csv(filename, header=0, sep=',',
                     engine='python', error_bad_lines=False)
    print(df.shape)
    df = df.fillna(' ')
    df['Name'] = df['givenName'].astype(
        str) + df['middleName'].astype(str) + df['surname'].astype(str)

    # print(df['Name'])
    return df


def populate_Trie(df):
    d = df.values.T
    t1 = Trie()
    t2 = Trie()
    t3 = Trie()
    t4 = Trie()
    for i in d[0]:
        t1.insert(str(i))
    for i in d[1]:
        t2.insert(str(i))
    for i in d[2]:
        t3.insert(str(i))
    for i in d[3]:
        t4.insert(str(i))

    return (t1, t2, t3, t4)


if __name__ == '__main__':

    df = loadData('data.csv')
    (t1, t2, t3, t4) = populate_Trie(df)
    print(get_best_matches(t1, 'Mah'))
    print(t1.search('Mahjabeen'))
