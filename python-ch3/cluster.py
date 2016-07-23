import scipy as sp

def tfidf(term, doc, docset):
    # not work
    tf = float(doc.count(term))/sum(doc.count(w) for w in docset)
    print(tf)

a, abb, abc = ['a'], ['a', 'b', 'b'], ['a', 'b', 'c']
D = [a, abb, abc]
print(tfidf('a', a, D))
