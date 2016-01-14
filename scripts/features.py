from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from scipy import sparse

import pandas as pd
import numpy as np

class FeatureTransformer(BaseEstimator):
	"""
	Encodes categorical features to numerical features
	"""

	def __init__(self, train, test):
		self.X = train
		self.X_test = test

	def get_feature_names(self):
		feature_names = []

		feature_names.extend(['blog_type', 'zodiac'])
		features_names.extend(self.truncated_svd.get_feature_names())

		return np.array(feature_names)

	def fit(self, X, y=None):
		self.fit_transform(X, y)

		return self

	def fit_transform(self, X, y=None):
		basic_features = self._process_basic_features(X)
		
		self.tfidf_vect = TfidfVectorizer(min_df=1, max_features=None, 
            strip_accents='unicode', analyzer='word',
            ngram_range=(1, 1), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
		
		tf_idf_features = self.tfidf_vect.fit_transform(X.blog_text)
		self.truncated_svd = TruncatedSVD(n_components=8)
		truncated_svd_features = self.truncated_svd.fit_transform(tf_idf_features)

		features = []
		
		features.append(basic_features)
		features.append(truncated_svd_features)

		features = np.hstack(features)
		
		return features

	def _process_basic_features(self, X):
		'Only process blog_type and zodiac'

		categorical_features = []

		for cat in ['blog_type', 'zodiac']:
			lbl = LabelEncoder()

			lbl.fit(pd.concat([self.X[cat], self.X_test[cat]], axis=0))

			categorical_features.append(lbl.transform(X[cat]))

		return np.array(categorical_features).T
		

	def transform(self, X):
		
		basic_features = self._process_basic_features(X)
		
		self.tfidf_vect = TfidfVectorizer(min_df=1, max_features=None, 
            strip_accents='unicode', analyzer='word',
            ngram_range=(1, 1), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
		
		tf_idf_features = self.tfidf_vect.fit_transform(X.blog_text)
		self.truncated_svd = TruncatedSVD(n_components=8)
		truncated_svd_features = self.truncated_svd.fit_transform(tf_idf_features)

		features = []
		
		features.append(basic_features)
		features.append(truncated_svd_features)

		features = np.hstack(features)
		
		return features
