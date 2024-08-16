import lxmls.readers.sentiment_reader as srs
import lxmls.classifiers.multinomial_naive_bayes as mnbb

# multinomial naive exercise

def mnb_exercise():

    scr = srs.SentimentCorpus("books")
    mnb = mnbb.MultinomialNaiveBayes()
    params_nb_sc = mnb.train(scr.train_X,scr.train_y)
    y_pred_train = mnb.test(scr.train_X,params_nb_sc)
    acc_train = mnb.evaluate(scr.train_y, y_pred_train)
    y_pred_test = mnb.test(scr.test_X,params_nb_sc)
    acc_test = mnb.evaluate(scr.test_y, y_pred_test)
    print("Multinomial Naive Bayes Amazon Sentiment Accuracy train: %f test: %f"%(acc_train,acc_test))

if __name__ == "__main__":
    print("hi")

    mnb_exercise()