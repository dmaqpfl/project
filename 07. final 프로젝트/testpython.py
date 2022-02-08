import test_func as tf

def summarize_last(doc) :
    sentences = tf.text2sentences(doc)
    if sentences != False : 
        nouns = tf.get_nouns(sentences)
        sent_graph = tf.build_sent_graph(nouns)
        sent_rank_idx = tf.get_rank(sent_graph)
        sorted_sent_rank_idx = sorted(sent_rank_idx, key=lambda k : sent_rank_idx[k], reverse=True)
        return tf.summarize_no(sentences, sorted_sent_rank_idx)
    else :
        return sentences







