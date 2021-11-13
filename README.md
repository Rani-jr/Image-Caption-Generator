# Image-Caption-Generator

Generating captions is a challenging problem: We should convert a given input image into a natural language description ( caption )

On one hand, it requires computer vision to understand the content of the image and, on the other hand, a language model from the field of NLP to turn the understanding of the image into words in order. Our task could be divided into two major parts:

* Image based model — Extracts the features of our image.
* Language based model — which translates the features and objects extracted by our image based model to a natural sentence.

## Testing

![img1](https://user-images.githubusercontent.com/88405252/141661885-89b17a21-a4b1-4c25-a68d-836268590ad9.JPG)

![img2](https://user-images.githubusercontent.com/88405252/141661888-d9a23429-f841-4a0b-bbc2-a735db06da2d.JPG)

These captions are quite accurate, but there is still room for improvement.

## Conclusion

We have implemented a CNN-LSTM model (merged-model) for building an Image Caption Generator. A CNN-LSTM architecture has wide-ranging applications which include use cases in Computer Vision and Natural Language Processing domains.
