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
    *   $$S = \frac{max_{real}-min_{real}}{max_q-min_q}$$
    *   $$Z = \lfloor\frac{0-min_{real}}{S}\rceil$$
*   위 수식에 따라 ``register_buffer``에 S, Z를 가지고 있는데, 이것을 int_only model로 내보낼 방법을 잘 모르겠음.

*   정수 텐서를 곱하는 과정의 Pseudo Code
```
activation: torch.tensor, dtype=torch.int8
weight: torch.tensor, dtype=torch.int8
bias: torch.tensor, dtype=torch.int32

real_product = torch.matmul(real_activation, real_weight.t()) + real_bias

a_16 = activation.to(torch.int16)
w_16 = weight.to(torch.int16)

row_sum = torch.sum(a_16, dim=1)
col_sum = torch.sum(w_16, dim=0)

product = (a_16 * w_16.t()).to(torch.int32)
        + hidden_dim * a_16.zero_point * w_16.zero_point
        - a_16.zero_point * col_sum     # is it sure?
        - w_16.zero_point * row_sum     # is it sure?

z3 = zeropoint of output activation
s3 = scale of output activation
M = a_16.scale * w_16.scale / s3

product = z3 + int_only_multiplication(M, product)
        + bias

product = product.to(torch.int8)
output = Quantized_ReLU6(product )

```