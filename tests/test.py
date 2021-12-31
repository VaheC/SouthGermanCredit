from woe_enc import get_feat_power, WoeEncoder
import pandas as pd


def test_get_feat_power1():
	assert  get_feat_power(0.7) == 'Suspicious'

def test_get_feat_power2():
	assert  get_feat_power(0.04) == 'Weak predictors'

def test_WoeEncoder():
	temp_X = pd.DataFrame({
	         'feat1': ['A', 'B', 'D', 'A', 'A', 'A', 'B', 
	                   'D', 'D', 'C', 'A', 'B', 'C', 'C', 'B'],
	         'feat2': ['D', 'D', 'D', 'K', 'K', 'G', 'G', 'D', 
	                   'K', 'G', 'G', 'K', 'D', 'G', 'K']
	})

	temp_y = pd.Series(
	         [1, 0, 1, 1, 0, 1, 0, 0, 
	          1, 0, 0, 1, 1, 0, 1],
	         name='target'
	)

	expected_X = pd.DataFrame({
	             'feat1': [0.271933715483642,
	                       -0.133531392624523,
	                       0.559615787935423,
	                       0.271933715483642,
	                       0.271933715483642,
	                       0.271933715483642,
	                       -0.133531392624523,
	                       0.559615787935423,
	                       0.559615787935423,
	                       -0.826678573184468,
	                       0.271933715483642,
	                       -0.133531392624523,
	                       -0.826678573184468,
	                       -0.826678573184468,
	                       -0.133531392624523]
	})

	temp_enc = WoeEncoder()
	temp_enc.fit(temp_X, temp_y)
	pd.testing.assert_frame_equal(temp_enc.transform(temp_X), expected_X)