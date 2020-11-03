import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, plot_confusion_matrix
from sklearn.naive_bayes import ComplementNB
#Data reading
train_data_raw = np.genfromtxt('data/train_samples.txt', encoding='utf-8', delimiter='\t', dtype=None, names=('id', 'text'), comments=None)
train_labels_raw = np.genfromtxt('data/train_labels.txt', dtype=None, names=('id', 'label'))
validation_data_raw = np.genfromtxt('data/validation_samples.txt', encoding='utf-8', delimiter='\t', dtype=None, names=('id', 'text'), comments=None)
validation_labels_raw = np.genfromtxt('data/validation_labels.txt', dtype=None, names=('id', 'label'))
test_data_raw = np.genfromtxt('data/test_samples.txt', encoding='utf-8', delimiter='\t', dtype=None, names=('id', 'text'), comments=None)
#Data splitting
train_data = train_data_raw['text']
train_labels = train_labels_raw['label']
validation_data = validation_data_raw['text']
validation_labels = validation_labels_raw['label']
test_data = test_data_raw['text']
test_ids = test_data_raw['id']
#BoW creation an feature extraction, followed by Tfidf transformation
Vectorizer = TfidfVectorizer(lowercase=False, analyzer='char_wb', ngram_range=(4, 7))
train_features = Vectorizer.fit_transform(train_data, train_labels)
validation_features = Vectorizer.transform(validation_data)
test_features = Vectorizer.transform(test_data)
#Classification
Classifier = ComplementNB(6e-3)
np.savetxt('params.txt', fmt='%s', X=[Classifier.get_params()])
Classifier.fit(train_features, train_labels)
validation_predictions = Classifier.predict(validation_features)
test_predictions = Classifier.predict(test_features)
#Calculating validation data metrics
validation_accuracy = accuracy_score(validation_labels, validation_predictions)
np.savetxt('results/accuracy_nb.txt', [validation_accuracy], fmt='%s', header='accuracy', comments='')
print('Accuracy:')
print('Source:' + str(validation_accuracy))
validation_score = f1_score(validation_labels, validation_predictions, average='macro')
np.savetxt('results/score_nb.txt', [validation_score], fmt='%s', header='macro f1 score', comments='')
print('Macro F1 score:')
print('Source:' + str(validation_score))
#Generating test submission
submission = np.vstack((test_ids, test_predictions)).T
np.savetxt('submission_nb.csv', submission, fmt='%s', delimiter=',', header='id,label', comments='')

# For documentation

titles_options = [('Confusion matrix, without normalization', None, 'd', ''), ('Normalized confusion matrix', 'true', '.2g', '_normalized')]
np.set_printoptions(precision=2)

for title, norm, fmt, extra in titles_options:
    display = plot_confusion_matrix(Classifier, validation_features, validation_labels, normalize=norm, cmap=plt.cm.Blues, values_format=fmt)
    display.ax_.set_title(title)
    print(title)
    print(display.confusion_matrix)
    plt.savefig('Plots/confusion_matrix' + extra)
    plt.close()

plt.figure(figsize=(10, 6))
plt.title('Evolution of Macro F1 Score for different alpha values')
plt.xlabel('Alpha')
plt.ylabel('Macro F1 Score')
alphas = np.linspace(0.001, 0.1, 100)
scores = np.array([f1_score(validation_labels, ComplementNB(alpha=alpha).fit(train_features, train_labels).predict(validation_features), average='macro') for alpha in alphas])
plt.plot(alphas, scores)
plt.plot([alphas.min(), alphas[scores.argmax()]], [scores.max(), scores.max()], color='g')
plt.xticks(np.append(plt.xticks()[0], alphas[scores.argmax()]))
plt.yticks(np.append(plt.yticks()[0], scores.max()))
plt.plot([alphas[scores.argmax()], alphas[scores.argmax()]], [round(scores.min(), 3), scores.max()], color='g')
plt.xlim(alphas.min(), alphas.max())
plt.ylim(round(scores.min(), 3), round(scores.max(), 3))
plt.savefig('Plots/alpha.png')
plt.close()
