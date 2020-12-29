# 3D-Human-Pose-Estimator
Cute pipeline for 3D human pose keypoints estimation.


Just match the human pose keypoints estimation with the monocular depth estimation. A little bit of Open3D geometry magic. And that's all.
The 3D box estimation comes from the limits of the keypoints already estimated in 3D.

## Results wearing a dressing gown
![](ezgif.com-video-to-gif.gif)
Not sure how is gonna behave wearing any other clothes. I haven't tried it yet.


## Stuff to improve

- [x] 3D video results visualization.
- [ ] A human segmentation to approximate the keypoints to the body to not have outliers estimated to the background.
- [ ] Try it all in an end-to-end approach.
- [ ] Too slow!
