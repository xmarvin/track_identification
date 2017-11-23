## Detector

Class **Detector** is designed to save movement information for each player in detection database.
There are missing data for some frames, and some rows contain incorrect data. Wrong jersey number or wrong team.


There are missing data for some frames, and some rows contain incorrect data in detection data.

Method *add_confidence_metric* based on idea that player's position can't change too much in 1/25 second.

Method *fill_tracks_linear* fills empty cells, based on naive suggestion that player was moving from previous  known position to next directly.

### Further ideas:
 Put distance between neighboring positions into the confidence calculation.