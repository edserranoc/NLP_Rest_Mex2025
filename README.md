# Rest-Mex 2025: Research on Sentiment Analysis Task for Mexican Tourist Texts (FrogCode Team) 🇲🇽 

The goal of this task is to analyze TripAdvisor Spanish-language reviews and classify them based on three key aspects:

-  sentiment polarity
-  type of site
-  associated Pueblo Mágico.

Each review contains valuable information about a traveler's experience, and our objective is to extract meaningful insights from it. First, we need to determine the sentiment polarity of the review by assigning it a rating from 1 (very negative) to 5 (very positive), based on the original score given by the tourist. This will help in understanding overall visitor satisfaction.

Next, we classify the review according to the type of site being reviewed. The review could describe a hotel, a restaurant, or an attraction, and this categorization is based on contextual keywords and available metadata.

## Training Dataset

- File: Rest-Mex_2025_Train.csv
- Size: 208,051 instances (70% of the original dataset)
- Columns:

  - 📌 Title: The title given by the tourist to their opinion (Text).
  - 📝 Review: The full review written by the tourist (Text).
  - 🎭 Polarity: The sentiment polarity of the review (1 to 5).
  - 📍 Town: The town where the review is focused (Text).
  - 🌎 Region: The Mexican state where the town is located (Text). This feature is not for classification but can provide additional information.
  - 🍽️ Type: The category of the reviewed place (Hotel, Restaurant, Attractive).

To access the data, I recommend registering for the contest in [Rest-Mex 2025](https://sites.google.com/cimat.mx/rest-mex-2025/).

## Citing

If you use NLP project in your research, please use the following citation:

    @misc{...,
      title        = {...},
      month        = ...,
      year         = 2025,
      doi          = {...},
      url          = {...}
    }

## License

This project is licensed under the GPL-3.0 license. See the LICENSE file for details.