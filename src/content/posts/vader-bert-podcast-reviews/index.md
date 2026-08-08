---
title: "Comparing the Sentiment of Reviews and Ratings, with VADER and BERT"
description: "In a previous post we trained a recommender on a million ratings from Apple Podcasts. However, we didn't use the content of the reviews, which are an additional source of signal of user preference. Some of that signal can be extracted using sentiment analysis. In this post we will do so using two methods: VADER and BERT."
date: 2022-10-21
notebook: "notebooks/2022-10-21-vader-bert-podcast-reviews.ipynb"
archived: true
---

In another notebook we trained a recommender using collaborative filtering on a million ratings from Apple Podcasts. However, we didn't use the **content of the reviews**, which are an additional source of signal of user preference. Some of that signal can be extracted using **sentiment analysis** and could then be used to train a recommender system.

The sentiment of each review is (of course) highly correlated with the rating given by the user, but **this correlation is not absolute**. For example, there are some 1 star ratings for which the review text nonetheless clearly reflects a positive user preference, but the user rated it with 1 star to bring attention to some complaint: issues with the sound, or even that no new episodes have been released in a while. The recommender which is only trained on the ratings will miss these distinctions.

We will **compare two different sentiment analysis techniques**.

### VADER
First we look **VADER**. This method associates a sentiment score to each text, which ranges from -1 for very negative text to 1 for very positive text (there are multiple scores but we use the compound score). To classify the reviews by sentiment we will need to set thresholds for this score.

VADER consists of a bag of words approach *modified* by some heuristic rules.  The *bag of words* part refers to getting the score of a review simply by *adding* the scores of the individual words. Note that this approach would disregard the order of the words, so we can think of the words as being randomly shuffled in a metaphorical bag (of words). The problem with such a simplistic approach is that the order of the words actually matter quite a bit. This is why VADER adds some useful heuristics which take into account the order of the words to *some* extent. For example, "not" appearing closely before a word inverts the polarity of that word. The rules are explained in the original [paper](http://eegilbert.org/papers/icwsm14.vader.hutto.pdf), which is very well written and worth a read.

### BERT

The other method we will use is based on the newer and very popular BERT transformer (clearly ML researchers love puns). More precisely we will use **distilBERT**, a smaller version which is almost as precise but more efficient in both time and memory thanks to [knowledge distillation](https://en.wikipedia.org/wiki/Knowledge_distillation). What makes BERT so popular is that it was of the first large language model that was made widely accessible for people to fine-tune for their own NLP tasks. This allows us to take advantage of the enormous resources spent by Google in training BERT with general text data and simply fine-tune it for our particular use case in just hours or even minutes on a single GPU.

On Hugging Face there is a distilBERT transfomer which has already been fine-tuned for the task of sentiment analysis. They used a variation of the [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/) (SST), called **SST2**. SST consists of sentences from movie reviews which have been annotated by human judges (giving sentiment scores between 0 and 1 with a slider). In the SST2 version the labels are binary (0 or 1) instead of floats.

In this notebook we will compare VADER to the distilBERT model fine-tuned on SST2. In a *separate* notebook we will **fine-tune the original pretrained distilBERT** ourselves on this podcast reviews dataset.

Fine-tuning the model ourselves will result in significantly better predictions of the sentiment, or at least the sentiment reflected by the ratings (which are our labels for training).

On the flip side, using the model fine-tuned on SST2 allows us to explore the sentiment associated with the various reviews *independently* from the ratings. We mentioned above that some 1 star ratings are actually just constructive feedback and the review content itself is mostly positive. The model trained on a different dataset (like SST2) is more likely classify those as being positive despite the low rating. In contrast, the model we train on the podcast reviews will learn to correlate the sentiment predictions with the star ratings as much as possible.

## 1. Sentiment Analysis with VADER

VADER relies on a lexicon of words, each with an associated polarity score. There are actually multiple scores but we will use **compound score**, which ranges from -1 (very negative) to 1 (very positive), and can be anywhere in between depending on the intensity of the sentiment. As mentioned in the introduction, the score of a sentence is roughly given by adding the scores of the individual words up, *except* that there are some heuristic rules. One such rule is inverting the score of a word if it is preceded by "not". Considering how simple this method is, it works surprisingly well. One helpful feature is that **the sentiment lexicon even contains emojis**, which are used in many reviews.

### 1.1 Load and Explore Data

In this section we:
- Load the data from an SQLite file.
- Compute the VADER polarity score of the reviews.
- Visualize the score distribution.
- "Demojize" the reviews (for distilBERT).
- Save the polarity scores and demojized reviews with Pickle.

First we need to load the data and save it in a Pandas DataFrame.

```python
with sqlite3.connect(os.path.join(PATH, 'data', 'database.sqlite')) as con:
  get_reviews = """SELECT author_id AS user_id, p.podcast_id, r.title, r.content, rating, p.title AS name, created_at
                              FROM podcasts p
                              INNER JOIN reviews r
                              USING(podcast_id)
                              """
  reviews_raw = pd.read_sql(get_reviews, con, parse_dates='created_at')
```

Next we will compute the polarity score for each review. We use the SentimentIntensityAnalyzer from vaderSentiment. The polarity score has multiple components but we only need the **compound score**.

```python
def polarity_score(text):
  sia = SentimentIntensityAnalyzer()
  return sia.polarity_scores(text)['compound']
```

```python
polarity_score('I did not hate the movie.')
```

    0.4585

It even works on emojis! This is actually relevant here because some podcast reviews contain emojis.

```python
polarity_score('😊')
```

    0.7184

Two smiley faces are better than one:

```python
polarity_score('😊😊')
```

    0.9001

To compute one polarity score per review we will concatenate the title and the body of the review:

```python
reviews_raw['review'] = reviews_raw['title'] + '. ' + reviews_raw['content']
```

Now we compute the polarity score for all one million reviews, which takes a few minutes!

```python
reviews_raw['polarity score'] = reviews_raw['review'].apply(polarity_score)
```

To feed the reviews to distilBERT later we need to convert emojis to text. Otherwise, they will be tokenized as 'unkown' and the information will be lost. We use the `emoji` Python package.

```python
reviews_raw['demojized review'] = reviews_raw['review'].apply(emoji.demojize)
```

We pickle the reviews dataframe to use in other notebooks. It also makes our life easier because we don't have to repeat the computation of the polarity score (which takes over 15 minutes) every time we start a new session.

```python
reviews_raw.to_pickle(os.path.join(PATH, 'data', 'reviews_raw_sentiment.pkl'))
```

```python
reviews_raw = pd.read_pickle(os.path.join(PATH, 'data', 'reviews_raw_sentiment.pkl'))
```

```python
reviews_raw.head(2)
```

  <div id="df-ec465d5c-c0e1-416f-8af1-653192915dfa">
    <div class="colab-df-container">
      <div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>podcast_id</th>
      <th>title</th>
      <th>content</th>
      <th>rating</th>
      <th>name</th>
      <th>created_at</th>
      <th>review</th>
      <th>polarity score</th>
      <th>demojized review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>F7E5A318989779D</td>
      <td>c61aa81c9b929a66f0c1db6cbe5d8548</td>
      <td>really interesting!</td>
      <td>Thanks for providing these insights.  Really e...</td>
      <td>5</td>
      <td>Backstage at Tilles Center</td>
      <td>2018-04-24 12:05:16-07:00</td>
      <td>really interesting!. Thanks for providing thes...</td>
      <td>0.9109</td>
      <td>really interesting!. Thanks for providing thes...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>F6BF5472689BD12</td>
      <td>c61aa81c9b929a66f0c1db6cbe5d8548</td>
      <td>Must listen for anyone interested in the arts!!!</td>
      <td>Super excited to see this podcast grow. So man...</td>
      <td>5</td>
      <td>Backstage at Tilles Center</td>
      <td>2018-05-09 18:14:32-07:00</td>
      <td>Must listen for anyone interested in the arts!...</td>
      <td>0.9739</td>
      <td>Must listen for anyone interested in the arts!...</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ec465d5c-c0e1-416f-8af1-653192915dfa')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-ec465d5c-c0e1-416f-8af1-653192915dfa button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ec465d5c-c0e1-416f-8af1-653192915dfa');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>

Having a look at the reviews, we see that VADER does catch some reviews that show the **user actually likes the podcast but has some minor complaint** to make. In that sense, one could use VADER to get the true user preference, which the rating is not reflecting correctly in those cases. 

The following is an example of such a review. The user clearly likes the podcast yet left a 1 star rating.

```python
reviews_raw.loc[9, 'content'], reviews_raw.loc[9, 'rating']
```

    ('Great podcast, but the editors turn the volume down for the talks. The intros are loud, then you have to crank up the volume for the talk.',
     1)

However, VADER exhibits a **positivity bias** and classifies many clearly negative reviews as positive. Because of this, it is probably not precise enough to give a useful signal of user preference in addition to the user rating. We will see that the sentiment predicted by the distilBERT model is much accurate.

Below we visualize the distribution of the VADER sentiment score for negative (1 and 2 star), neutral (3 star) and positive (4 and 5 star) ratings.

```python
def plot_histograms_by_sentiment(reviews, column_name):
  fig, axs = plt.subplots(1, 3, figsize=(12, 4))
  sns.histplot(
      reviews[reviews['rating'].isin([1, 2])][column_name],
      ax=axs[0],
      bins=30,
      kde=True,
  )
  sns.histplot(
      reviews[reviews['rating'] == 3][column_name],
      ax=axs[1],
      bins=30,
      kde=True,
  )
  sns.histplot(
      reviews[reviews['rating'].isin([4, 5])][column_name],
      ax=axs[2],
      bins=30,
      kde=True,
  )
  axs[0].set_title('1 and 2 stars')
  axs[1].set_title('3 stars')
  axs[2].set_title('4 and 5 stars')
  fig.tight_layout()
plot_histograms_by_sentiment(reviews_raw, 'polarity score')
```

    
![](./output_25_0.png)
    

The histograms clearly show a positivity bias. We see that even for negative ratings the mean sentiment score is just over 0:

```python
neg_mean = reviews_raw[reviews_raw['rating'].isin([1, 2])]['polarity score'].mean()
neut_mean = reviews_raw[reviews_raw['rating'] == 3]['polarity score'].mean()
pos_mean = reviews_raw[reviews_raw['rating'].isin([4, 5])]['polarity score'].mean()
print(
  f'The mean VADER compound score for 1 and 2 star reviews is {neg_mean:.2}\n'
  f'The mean VADER compound score for 3 star reviews is {neut_mean:.2}\n'
  f'The mean VADER compound score for 4 and 5 star reviews is {pos_mean:.2}'
)
```

    The mean VADER compound score for 1 and 2 star reviews is 0.017
    The mean VADER compound score for 3 star reviews is 0.37
    The mean VADER compound score for 4 and 5 star reviews is 0.79

The peaks at 0 are probably reviews for which VADER can't actually identify the sentiment. Regarding the 0 scores, a word of caution: The histograms can be misleading! The reviews with score 0 seem to be a large proportion for 1 and 2 star ratings, and certainly seem to comprise much smaller proportions for the other rating values. However, we see below that the differences are actually not as dramatic as they might look in the histograms: reviews with score 0 are approximately $4\%$ for negative ratings, $3\%$ for neutral ratings, and $2\%$ for positive ratings.

```python
reviews_raw.groupby('rating').apply(lambda df: (df['polarity score'] == 0).mean())
```

    rating
    1    0.052295
    2    0.035327
    3    0.032470
    4    0.025381
    5    0.023363
    dtype: float64

### 1.2 Clean Data
Some reviews appear to be spam, which is why we will remove reviews by users with suspiciously high review counts. We will also exclude some podcasts for kids because a majority of the "reviews" for those podcasts aren't actually reviews. Instead, children appear to be using the reviews as a forum in which to post jokes.

Additionally, we are writing two functions to convert both VADER polarity scores and ratings into **sentiment classes**. We will contemplate two possibilities:
- Three classes: 0 (negative), 1 (neutral) and 2 (positive).
- Binary case: 0 (negative) and 1 (positive).

The functions below can handle either case.

```python
kids_podcasts = ['Wow in the World', 'Story Pirates', 'Pants on Fire', 'The Official Average Boy Podcast', 'Despicable Me', 'Rebel Girls', 'Fierce Girls', 'Like and Subscribe: A podcast about YouTube culture', 'The Casagrandes Familia Sounds', 'What If World - Stories for Kids', 'Good Night Stories for Rebel Girls', 'Gird Up! Podcast', 'Highlights Hangout', 'Be Calm on Ahway Island Bedtime Stories', 'Smash Boom Best', 'The Cramazingly Incredifun Sugarcrash Kids Podcast']

def remove_spammers(reviews, max_reviews=135):
    'Remove users with suspiciously high review count.'
    mask = reviews.groupby('user_id')['podcast_id'].transform('count') <= max_reviews
    return reviews[mask]

def rating_to_sentiment(ratings, neutral=True):
  sentiments = np.zeros(ratings.shape)
  sentiments[ratings == 3] = 1 if neutral else 0
  sentiments[ratings > 3] = 2 if neutral else 1
  return sentiments

def vader_score_to_sentiment(polarity_scores, neg_threshold=0.4, pos_threshold=0.75):
  assert neg_threshold <= pos_threshold
  sentiments = np.zeros(polarity_scores.shape)
  sentiments[polarity_scores > neg_threshold] = 1
  if pos_threshold > neg_threshold: # otherwise there is no neutral class
    sentiments[polarity_scores > pos_threshold] = 2
  return sentiments
```

```python
reviews_raw['VADER sentiment'] = vader_score_to_sentiment(reviews_raw['polarity score'])
reviews_raw['sentiment'] = rating_to_sentiment(reviews_raw['rating'])
reviews_raw['binary sentiment'] = rating_to_sentiment(reviews_raw['rating'], neutral=False)
```

Note that in addition to cleaning the data we are taking a sample consisting of 100,000 reviews. This makes the data more manageable while still being a large enough dataset to be representative when we evaluate our sentiment classifiers. On top of that, **we sample the data in such a way that each star rating is represented equally**, to make sure that classification accuracy isn't skewed in favor of positive ratings, which constitute over $90\%$ of the original dataset.

```python
reviews_raw['sentiment'].value_counts() / reviews_raw['sentiment'].count()
```

    2.0    0.905482
    0.0    0.071179
    1.0    0.023339
    Name: sentiment, dtype: float64

Now we are finally ready to do the cleaning and take a 100,000 reviews sample with equal ratings representation.

```python
 reviews = (
     reviews_raw.query("name not in @kids_podcasts")
                .pipe(remove_spammers)
                .groupby('rating')
                .apply(lambda df: df.sample(n=20000))
                .sample(frac=1)
                .reset_index(drop=True)
)
```

### 1.3 Results for VADER Classification into Negative, Neutral, and Positive

We used the VADER score to classify reviews into those three classes based on two thresholds (which we tuned by hand to maximize accuracy).

The ratings were used as the ground truth sentiment, where 1 and 2 star ratings correspond to negative, 3 star ratings to neutral, and 4 and 5 star ratings to positive.

The following is the confusion matrix for the whole (raw) dataset.

```python
pd.crosstab(reviews_raw['VADER sentiment'], reviews_raw['sentiment'])
```

  <div id="df-df6e628f-e140-40ac-9b31-26524e3fc533">
    <div class="colab-df-container">
      <div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>sentiment</th>
      <th>0.0</th>
      <th>1.0</th>
      <th>2.0</th>
    </tr>
    <tr>
      <th>VADER sentiment</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>45084</td>
      <td>9247</td>
      <td>79046</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>11054</td>
      <td>4525</td>
      <td>116317</td>
    </tr>
    <tr>
      <th>2.0</th>
      <td>13931</td>
      <td>9203</td>
      <td>695998</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-df6e628f-e140-40ac-9b31-26524e3fc533')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-df6e628f-e140-40ac-9b31-26524e3fc533 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-df6e628f-e140-40ac-9b31-26524e3fc533');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>

```python
accuracy_score(reviews_raw['sentiment'], reviews_raw['VADER sentiment'])
```

    0.757418948501887

The accuracy is relatively high but this can be misleading because in the original dataframe `reviews_raw` over $90\%$ of ratings are positive.

The recall shows that the classification is no better than chance when restricted to neutral reviews (if we picked a rating at random we would get 3 stars, i.e. neutral, $20\%$ of the time, although the fact that the recall is $19.7\%$ is probably a coincidence).

```python
recall_score(reviews_raw['sentiment'], reviews_raw['VADER sentiment'], average=None)
```

    array([0.64342291, 0.19695321, 0.78082617])

The accuracy on the cleaned data in `reviews` is less misleading because we made sure that all ratings are equally represented with 20,000 reviews each:

```python
pd.crosstab(reviews['VADER sentiment'], reviews['sentiment'])
```

  <div id="df-89158054-9c1f-4fee-b046-228a9f8952a5">
    <div class="colab-df-container">
      <div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>sentiment</th>
      <th>0.0</th>
      <th>1.0</th>
      <th>2.0</th>
    </tr>
    <tr>
      <th>VADER sentiment</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>24394</td>
      <td>8080</td>
      <td>5497</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>6680</td>
      <td>3930</td>
      <td>6007</td>
    </tr>
    <tr>
      <th>2.0</th>
      <td>8926</td>
      <td>7990</td>
      <td>28496</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-89158054-9c1f-4fee-b046-228a9f8952a5')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-89158054-9c1f-4fee-b046-228a9f8952a5 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-89158054-9c1f-4fee-b046-228a9f8952a5');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>

We see that on `reviews` the accuracy is much lower but the recall is similar (it is a little lower but that might change if we choose different thresholds for the VADER score).

```python
accuracy_score(reviews['sentiment'], reviews['VADER sentiment'])
```

    0.5682

```python
recall_score(reviews['sentiment'], reviews['VADER sentiment'], average=None)
```

    array([0.60985, 0.1965 , 0.7124 ])

### 1.4 Optimal Threshold for VADER and Binary Sentiment

From now on we will consider a **binary** classification problem with the classes negative and positive, i.e. **discarding the neutral category**. We do this because the fine-tuned distilBERT model we are using is only a binary classifier. Note: In a separate notebook we will train distilBERT to predict the ratings, which would allow us to have a neutral class or even just 5 classes (the ratings themselves).

It seems clear that reviews with 1 or 2 stars should be considered negative and reviews with 4 and 5 stars positive. The question is *how to classify the 3 star reviews*. While VADER mostly gives them positive scores, we will see that the distilBERT model actually mostly classifies them as negative. From reading some of the 3 star reviews it does appear that the distilBERT model is right and we already noted that VADER has a positivity bias.

To classify the reviews into two classes using VADER we just have a single threshold: everything to the left of it is negative and everything to the right positive. With the following function we will find the **threshold resulting in the highest possible classification accuracy**, given a list of VADER scores and corresponding ground truth sentiments. *This is just intended as a baseline for the distilBERT model and is not a principled way to tune VADER, since this threshold probably has a high variance and we are overfitting on our training set.*

```python
def find_best_split(reviews, score_col='polarity score', sentiment_col='binary sentiment'):
  sorted_df = (
    reviews.sort_values(by=score_col)
           [[score_col, sentiment_col]]
  )
  scores = sorted_df[score_col]
  sentiments = sorted_df[sentiment_col]
  correct_class = max_correct = sentiments.sum()
  optimal_thresh = prev_score = -1
  count = 0
  for score, sentiment in zip(scores, sentiments):
    if sentiment == 0:
      correct_class += 1
    else:
      if score != prev_score and correct_class > max_correct:
        optimal_thresh = prev_score
        max_correct = correct_class
      correct_class -= 1
    prev_score = score
  if correct_class > max_correct:
    optimal_thresh = score
    max_correct = correct_class
  return {'threshold': optimal_thresh, 'accuracy': max_correct / scores.size}
```

First we will define 3 star ratings as negative (in fact, we already did this when we computed the 'binary sentiment' column above).

```python
best_split = find_best_split(reviews)
best_split
```

    {'threshold': 0.7945, 'accuracy': 0.71772}

Next let's think of 3 star ratings as positive instead (we call it *alternative binary sentiment*).

```python
reviews['alt binary sentiment'] = reviews['rating'].map({1:0,2:0,3:1,4:1,5:1})
print(find_best_split(reviews, sentiment_col='alt binary sentiment'))
reviews = reviews.drop(columns='alt binary sentiment');
```

    {'threshold': 0.296, 'accuracy': 0.70864}

Considering 3 star reviews to be positive instead of negative made virtually no difference to the accuracy. This is a little surprising because VADER tends to give 3 star reviews positive sentiment scores. The reason that the accuracy doesn't improve is that a lower threshold results in a lower recall for negative ratings, on which VADER also has a positivity bias.

Below we compute the recall. It isn't great but not terrible either considering the simplicity of the VADER method and the difficulty of the task. However, distilBERT will do better and without the need to fine-tune it on our data (although, as I mentioned, we will fine-tune it in a separate notebook and the accuracy will improve significantly).

```python
recall_score(reviews['polarity score'] >= best_split['threshold'], reviews['binary sentiment'], average=None)
```

    array([0.77318932, 0.6405552 ])

## 2. BERT for Sentiment Classification

As mentioned in the introduction, we are using a distilBERT model [fine-tuned](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) on the SST2 dataset consisting of sentences from movie reviews.

```python
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_SST)
bert_model = AutoModelForSequenceClassification.from_pretrained(FINETUNED_SST)
```

Before being fed to the transformer we need to tokenize the text. Tokens often correspond to full words but can also correspond to parts of words (this happens for rare words) or symbols like punctuation.

The maximum length the transformer can handle is 512 so we will have to clip particularly long reviews. In fact, we will set a lower maximum than that to improve performance. A single long review would mean we have to make the whole batch longer (the lengths of the samples in the batch must agree and the shorter ones are filled with placeholder tokens) and this uses more memory on the GPU and requires more computations.

The cutoff should be larger than the length of the overwhelming majority of reviews, to make sure it has negligible effect on the precision of the model. To determine this cutoff we will plot the length distribution.

```python
token_lengths = np.array([len(tokenizer.encode(s, truncation=True, max_length=512)) for s in reviews['demojized review']])
sns.histplot(token_lengths, kde=True)
plt.xlabel('Token count for review');
```

    
![](./output_60_0.png)
    

```python
f'Just {(np.array(token_lengths) >= 256).mean()*100:.2} percent of the reviews have a length over 256'
```

    'Just 2.7 percent of the reviews have a length over 256'

Now we take the **demojized reviews** from the `reviews` dataframe, **tokenize** them with a **maximum length of 256 tokens** and create a dataloader which will feed the tokenized samples in batches of size 32 to the distilBERT classifier.

```python
def tokenize_function(data, tokenizer, max_length=256):
    return tokenizer(data['demojized review'], truncation=True, max_length=max_length)

dataset = Dataset.from_dict(reviews[['demojized review']])

tokenized_dataset = (
    dataset.map(partial(tokenize_function, tokenizer=tokenizer), batched=True)
           .remove_columns(['demojized review'])
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

dataloader = DataLoader(
    tokenized_dataset, batch_size=32, collate_fn=data_collator
)
```

The distilBERT model outputs the logits for the targets 0 (negative) and 1 (positive). The following function evaluates the model on a dataloader and returns an array of **probabilities** for the reviews fed through the dataloader being **positive**.

```python
def get_probs(model, dataloader):
  probs = []
  model = model.to(device)
  model.eval()
  m = nn.Softmax(dim=1)
  for batch in tqdm(dataloader):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
      outputs = model(**batch)
      logits = outputs.logits
      probs += m(logits)[:,1].tolist()
  return np.array(probs)
```

```python
reviews['BERT probs'] = get_probs(bert_model, dataloader)
```

## 3. Comparing VADER and BERT

We see in the following histograms that distilBERT classifies most 3 star ratings as negative. **This is interesting because VADER does the complete opposite**, assigning overwhelmingly positive scores to 3 star reviews.

Something else to note is that this model is **very confident in its predictions**, with two sharp peaks around 0 and 1 but very little in between. We can see a little less confidence for 3 star ratings. Those are the most mixed reviews in terms of sentiment and they do exhibit some more intermediate probability values than other star ratings. However, the VADER score does a much better job as a continuous measure of sentiment outside of 0 and 1. To be fair, the distilBERt classifier is intended to make correct binary predictions, not to quantify uncertainty.

```python
plot_histograms_by_sentiment(reviews, 'BERT probs')
```

    
![](./output_68_0.png)
    

Most times when VADER and distilBERT disagree, the latter is right. This is not surprising because distilBERT is a much more complicated and computation intensive technique.

The following is a typical example which has a very high VADER score yet very low BERT probability of being positive (and BERT is right).

```python
reviews.loc[945, ['title', 'content', 'rating', 'polarity score', 'BERT probs']]
```

    title                                              Used to be great
    content           Used to be a great comedy podcast, until, in s...
    rating                                                            1
    polarity score                                               0.9509
    BERT probs                                                 0.022963
    Name: 945, dtype: object

```python
reviews.loc[945, 'review']
```

    'Used to be great. Used to be a great comedy podcast, until, in someone\'s infinite wisdom, decided to replace the only talent on that "network".'

The reason the VADER score is so high for that review is that it contains many words with positive sentiment (great, wisdom, talent) and not really any words with negative sentiment (in isolation). The distilBERT model however is able to take into account the context of the whole sentence ("used to be", "the only talent").

Let's look at the reviews with high probability of being positive according to distilBERT but a very negative VADER score, and vice versa.

We see below that there are very few cases in the former category but many in the latter. The distilBERT model is usually right but certainly not every time.

Actually going through the reviews one gets the impression that those numbers underestimate how much better distilBERT is to VADER. In many cases the review sentiment is only loosely correlated with the rating. As such, some "misclassifications" by distilBERT could even be seen as additional signal to the ratings rather than mistakes.

```python
reviews.loc[(reviews['BERT probs'] > 0.95) & (reviews['polarity score'] < -0.9), 'rating'].value_counts()
```

    4    17
    3    17
    1    16
    2    13
    5    13
    Name: rating, dtype: int64

```python
reviews.loc[(reviews['BERT probs'] < 0.05) & (reviews['polarity score'] > 0.9), 'rating'].value_counts()
```

    3    1753
    2    1731
    1     871
    4     769
    5      74
    Name: rating, dtype: int64

Here are some 3 star reviews that distilBERT classifies as positive.

```python
reviews[(reviews['BERT probs'] > 0.99) & (reviews['rating'] == 3)][['title', 'content', 'rating', 'polarity score', 'BERT probs']].head(10)
```

  <div id="df-b5fce6e9-4284-4542-a08d-28e153e0adff">
    <div class="colab-df-container">
      <div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>content</th>
      <th>rating</th>
      <th>polarity score</th>
      <th>BERT probs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>50</th>
      <td>☺️</td>
      <td>Love the podcast!! You guys keep me entertained!!</td>
      <td>3</td>
      <td>0.9015</td>
      <td>0.999867</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Great content… Please work on format</td>
      <td>I love this podcast - the information is so va...</td>
      <td>3</td>
      <td>0.9890</td>
      <td>0.993949</td>
    </tr>
    <tr>
      <th>62</th>
      <td>Cool, but...</td>
      <td>Smart and funny pod. She gets a little condesc...</td>
      <td>3</td>
      <td>0.8422</td>
      <td>0.996883</td>
    </tr>
    <tr>
      <th>221</th>
      <td>conservatives might pass on this</td>
      <td>I've been listening to this podcast for years ...</td>
      <td>3</td>
      <td>0.3679</td>
      <td>0.995914</td>
    </tr>
    <tr>
      <th>459</th>
      <td>Hi it’s me and my family</td>
      <td>Lion and cat cat dog cat cat</td>
      <td>3</td>
      <td>0.0000</td>
      <td>0.998568</td>
    </tr>
    <tr>
      <th>466</th>
      <td>JUST WOW!</td>
      <td>It’s like making a murderer REVERSED, this sou...</td>
      <td>3</td>
      <td>0.9485</td>
      <td>0.998971</td>
    </tr>
    <tr>
      <th>470</th>
      <td>Another re-run?</td>
      <td>Are you guys ever going to get back to creatin...</td>
      <td>3</td>
      <td>0.3076</td>
      <td>0.998778</td>
    </tr>
    <tr>
      <th>473</th>
      <td>Keep it up</td>
      <td>Hey, I'm a follow controller I like where this...</td>
      <td>3</td>
      <td>0.3612</td>
      <td>0.998447</td>
    </tr>
    <tr>
      <th>542</th>
      <td>Overuse of “incredible”</td>
      <td>I love these people but every other word Mallo...</td>
      <td>3</td>
      <td>0.3818</td>
      <td>0.996635</td>
    </tr>
    <tr>
      <th>584</th>
      <td>my opinion</td>
      <td>I love this podcast but you guys sidetrack way...</td>
      <td>3</td>
      <td>0.3818</td>
      <td>0.991521</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-b5fce6e9-4284-4542-a08d-28e153e0adff')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-b5fce6e9-4284-4542-a08d-28e153e0adff button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-b5fce6e9-4284-4542-a08d-28e153e0adff');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>

On the other hand, here is an example of a 3 star review that distilBERT classifies as negative.

```python
reviews[(reviews['BERT probs'] < 0.01) & (reviews['rating'] == 3)][['title', 'content', 'rating', 'polarity score', 'BERT probs']].head(10)
```

  <div id="df-3f3cd571-33fb-4ece-8b62-e03c2f845809">
    <div class="colab-df-container">
      <div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>content</th>
      <th>rating</th>
      <th>polarity score</th>
      <th>BERT probs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>Used to be great, now is just okay</td>
      <td>I preferred the old format. Loved hearing from...</td>
      <td>3</td>
      <td>0.8151</td>
      <td>0.004110</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Okay...</td>
      <td>Cool I guess but I can't download any songs cu...</td>
      <td>3</td>
      <td>0.7391</td>
      <td>0.007136</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Maz</td>
      <td>Bring him back please.   Not a good move.  Los...</td>
      <td>3</td>
      <td>-0.1546</td>
      <td>0.005427</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Decent content</td>
      <td>Overall the content is good and timely, but th...</td>
      <td>3</td>
      <td>-0.7420</td>
      <td>0.000644</td>
    </tr>
    <tr>
      <th>22</th>
      <td>You should</td>
      <td>Do an interview with Not The Worst Show, it co...</td>
      <td>3</td>
      <td>0.7575</td>
      <td>0.001318</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Addictive but trashy</td>
      <td>I listened to most of the first season. It's a...</td>
      <td>3</td>
      <td>0.9111</td>
      <td>0.000854</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Basically two guys catching a buzz......</td>
      <td>These guys seem to have a grasp on what a beer...</td>
      <td>3</td>
      <td>0.3498</td>
      <td>0.000442</td>
    </tr>
    <tr>
      <th>76</th>
      <td>not as good anymore</td>
      <td>The episodes were never as long as I wanted th...</td>
      <td>3</td>
      <td>0.8948</td>
      <td>0.004767</td>
    </tr>
    <tr>
      <th>78</th>
      <td>Inconsistent</td>
      <td>The shows vary from top shelf to really weak d...</td>
      <td>3</td>
      <td>-0.5563</td>
      <td>0.000442</td>
    </tr>
    <tr>
      <th>80</th>
      <td>Enjoy...Except for the Recyling</td>
      <td>I subscrobe to the show and enjoy it quite a b...</td>
      <td>3</td>
      <td>0.3716</td>
      <td>0.006824</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-3f3cd571-33fb-4ece-8b62-e03c2f845809')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-3f3cd571-33fb-4ece-8b62-e03c2f845809 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-3f3cd571-33fb-4ece-8b62-e03c2f845809');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>

Looking at the reviews, there is a clear difference between the ones classified as positive and those classified as negative, even though all of them come with 3 star ratings. **This exemplifies one way in which the review sentiment can give us additional signal of user preference.**

Now let's look at 1 and 2 star rating reviews that distilBERT classifies as positive. We can see that many of them talk about how they *used to* love the show, which confuses the model. Others complain about politics. We will see that at least anecdotally the distilBERT we fine-tune on this dataset will do better on those types of reviews that are common within this dataset.

```python
reviews[(reviews['BERT probs'] > 0.99) & reviews['rating'].isin([1, 2])][['title', 'content', 'rating', 'polarity score', 'BERT probs']].head(10)
```

  <div id="df-87413c99-9f63-4b51-861d-c2a8bedd486c">
    <div class="colab-df-container">
      <div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>content</th>
      <th>rating</th>
      <th>polarity score</th>
      <th>BERT probs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>Why</td>
      <td>Why would you trust anything from MSNBC? They ...</td>
      <td>2</td>
      <td>0.7096</td>
      <td>0.990558</td>
    </tr>
    <tr>
      <th>122</th>
      <td>Lurched Right</td>
      <td>Good folks. What happened to them at Evergreen...</td>
      <td>2</td>
      <td>0.5794</td>
      <td>0.992818</td>
    </tr>
    <tr>
      <th>239</th>
      <td>Great Content but....</td>
      <td>I love the idea of this show and listen to oth...</td>
      <td>2</td>
      <td>0.9259</td>
      <td>0.998793</td>
    </tr>
    <tr>
      <th>255</th>
      <td>No more Monica</td>
      <td>I’m certain you are a good person, and I would...</td>
      <td>2</td>
      <td>0.9603</td>
      <td>0.998205</td>
    </tr>
    <tr>
      <th>285</th>
      <td>Wanted to like this...</td>
      <td>Yikes, combine cheerful ignorance with minimal...</td>
      <td>2</td>
      <td>0.7331</td>
      <td>0.996498</td>
    </tr>
    <tr>
      <th>418</th>
      <td>hope Mr Black acknowledges the REAL creators o...</td>
      <td>If you want to hear this done properly, listen...</td>
      <td>1</td>
      <td>0.8689</td>
      <td>0.993219</td>
    </tr>
    <tr>
      <th>492</th>
      <td>More of the Same Now</td>
      <td>I used to absolutely love Invisibilia.  The ep...</td>
      <td>1</td>
      <td>0.8858</td>
      <td>0.993986</td>
    </tr>
    <tr>
      <th>523</th>
      <td>Tip very intelligent butttttt!!!</td>
      <td>Tip you have to let your guests speak to have ...</td>
      <td>1</td>
      <td>0.9429</td>
      <td>0.995483</td>
    </tr>
    <tr>
      <th>551</th>
      <td>Well, So Much For That</td>
      <td>When I first saw this podcast, I was VERY exci...</td>
      <td>2</td>
      <td>0.9633</td>
      <td>0.995017</td>
    </tr>
    <tr>
      <th>713</th>
      <td>And That’s Why You’re Awesome</td>
      <td>I have been listening to ATWWD for over a year...</td>
      <td>2</td>
      <td>0.9714</td>
      <td>0.999869</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-87413c99-9f63-4b51-861d-c2a8bedd486c')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-87413c99-9f63-4b51-861d-c2a8bedd486c button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-87413c99-9f63-4b51-861d-c2a8bedd486c');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>

Finally these are some 4 and 5 star rating reviews that distilBERt classifies as negative. We can see that it is mostly 4 star reviews and furthermore they all seem to be complaining. It makes sense that distilBERT would classify them as negative.

```python
reviews[(reviews['BERT probs'] < 0.01) & reviews['rating'].isin([4, 5])][['title', 'content', 'rating', 'polarity score', 'BERT probs']].head(10)
```

  <div id="df-0316346f-58f5-4472-8a9e-9e1b1dfa8209">
    <div class="colab-df-container">
      <div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>content</th>
      <th>rating</th>
      <th>polarity score</th>
      <th>BERT probs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>Great but....</td>
      <td>The interview with that Nolan guy was too anno...</td>
      <td>4</td>
      <td>0.8916</td>
      <td>0.001526</td>
    </tr>
    <tr>
      <th>58</th>
      <td>Are u dumb</td>
      <td>Yo flip release podcast audio already facts!!!...</td>
      <td>4</td>
      <td>-0.3348</td>
      <td>0.001746</td>
    </tr>
    <tr>
      <th>61</th>
      <td>New listener</td>
      <td>People have tried to get me interested in podc...</td>
      <td>5</td>
      <td>0.6908</td>
      <td>0.001139</td>
    </tr>
    <tr>
      <th>106</th>
      <td>Up speak?!</td>
      <td>The show content is really interesting? But th...</td>
      <td>4</td>
      <td>0.3981</td>
      <td>0.000748</td>
    </tr>
    <tr>
      <th>222</th>
      <td>sound issues</td>
      <td>i love the words when the sound is good enough...</td>
      <td>4</td>
      <td>0.1862</td>
      <td>0.009301</td>
    </tr>
    <tr>
      <th>244</th>
      <td>Great storytelling</td>
      <td>I really like this podcast but I gotta admit t...</td>
      <td>4</td>
      <td>0.8589</td>
      <td>0.008216</td>
    </tr>
    <tr>
      <th>257</th>
      <td>Weird music over talking</td>
      <td>Jan. 7 episode has some weird loud music over ...</td>
      <td>4</td>
      <td>-0.5204</td>
      <td>0.006214</td>
    </tr>
    <tr>
      <th>318</th>
      <td>I do like this podcast...</td>
      <td>I really like this podcast, however, John has ...</td>
      <td>4</td>
      <td>0.6953</td>
      <td>0.001644</td>
    </tr>
    <tr>
      <th>333</th>
      <td>Bananas...NOOOOOwaaahhhh....</td>
      <td>Let me say, I absolutely love the MFM podcast ...</td>
      <td>4</td>
      <td>-0.8815</td>
      <td>0.002038</td>
    </tr>
    <tr>
      <th>442</th>
      <td>Hoping season 4 is a return to form</td>
      <td>Seasons 1 &amp; 2 were utter brilliance. Season 3 ...</td>
      <td>4</td>
      <td>0.3167</td>
      <td>0.004957</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-0316346f-58f5-4472-8a9e-9e1b1dfa8209')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-0316346f-58f5-4472-8a9e-9e1b1dfa8209 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-0316346f-58f5-4472-8a9e-9e1b1dfa8209');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
