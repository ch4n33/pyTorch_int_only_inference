#  Int Arithmetic Only Inference
[Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf) 논문을 pytorch로 구현

*  Fake Quantization을 Float Precision으로 수행한 뒤, 학습이 완료된 모델을 8bit(bias는 32bit)로 quantize하여 저장

*  quantization scheme
    *  $$q = S(r - Z)$$
*  fake quantization scheme
    *  $$q = \lfloor\frac{clamp(r;a,b)-a}{s(a,b,n)}\rceil s(a,b,n) + a$$
    *  $$s(a,b,n) := \frac{b-a}{n-1}$$

##  미구현
*   fake quantization schema에 따라 만들어진 모델을 quantize하고, intager only arithmetic을 적용하는 방법을 모르겠음.
