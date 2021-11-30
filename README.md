# 文本生成任务

### requirements
tensorflow >= 2.0

### 一、seq2seq_attention
采用常规encoder(gru)-decoder(gru)结合attenion方式
>输入
>>来源 不 明 但是 只是 感觉 太 赞 ＼ 春丽 ／

>输出
>>好 腿 法 ， 不 知道 站 的 稳发 。

>数据来源
>>github上获取到的，训练集1800000+，应该是微博数据，太大上传不了。

### 二、seq2seq_copynet
>输入形式
>> ```key:["类型", "版型", "材质", "风格", "图案", "图案", "图案", "衣样式", "衣领型"]  value: ["上衣", "h", "蚕丝", "复古", "条纹", "复古", "撞色", "衬衫", "小立领"]  ```

>输出形式
>> ```["小", "女人", "十足", "的", "条纹", "衬衣", "，", "缎面", "一点点", "的", "复古", "，", "还有", "蓝绿色", "这种", "高级", "气质", "复古", "色", "，", "真丝", "材质", "，", "撞色", "竖", "条纹", "特别", "的", "现代感", "味道", "，", "直", "h", "型", "的", "裁剪", "和", "特别", "的", "衣长", "款式", "，", "更加", "独立", "性格", "。", "双层", "小立领", "，", "更显", "脸型", "。"] ```

>数据来源
>>Long and Diverse Text Generation with Planning-based Hierarchical Variational Model

>copy_net来源
>>Incorporating Copying Mechanism in Sequence-to-Sequence Learning

>Encoder
>>key_embedding 与 value_embedding拼接 、采用常规gru，单层

>Decoder
>>采用常规gru，单层 、attention机制 、添加copy_net机制，能够解决OOV问题，更适用于实际场景

>Copy机制
>>copy机制最初是为了解决OOV问题，如当有一些专有名词不在你训练时的词表中时，那在生成时普通的seq2seq是无论如何也无法生成出该词，copyNet的encoder端与普通seq2seq一致，在decode端生成时，copy机制的原理是，让它以一定的概率为生成词，一定的概率为复制该词



### 三、seq2seq_pgn
>输入形式
>> 同二

>来源
>>Get To The Point: Summarization with Pointer-Generator Networks

>模型特色

1.类似copynet，解决OOV问题
2.Coverage mechanism缓解seq2seq生成中容易出现重复问题，使句子更加连贯

>训练方式

需要先用主函数训练好一个收敛的模型，然后再把covloss加上，做个finetune，不然的话效果还是不好
