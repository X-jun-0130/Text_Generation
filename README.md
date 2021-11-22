# 文本生成任务

### requirements
tensorflow >= 2.0

### 一、seq2seq_attention
采用常规encoder(gru)-decoder(gru)结合attenion方式

### 二、seq2seq_copynet
>Encoder
>>采用常规gru，单层

>Decoder
>>添加copy_net机制，能够结构OOV问题，更适用于实际场景
 
