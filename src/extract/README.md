# How to run

```python my_extract_features.py --input_file ./data/post_data.csv --bert_config_file ./uncased_base/bert_config.json --init_checkpoint ./uncased_base/bert_model.ckpt --output_dir ./data --vocab_file ./uncased_base/vocab.txt```

input file format:

# df

input data is a dataframe file which includes "content_txt" column, indicates the content of a post

     .   content_id     .                           content_txt                               
     ..................................................................................................
     .  200615215323    .    <A|925284419> I dont wanna spam the same info like were all seeing.$HH$$HH$Just here to say we told you so! Where are the non believers? All the people that said this was a shit stock? I dont see you in the comments ANYWHERE!🥴$HH$$HH$We like the stock! 👐💎🆘.	   
     .   2005054986     .    <A|950181551> ETC a coin that was hacked and will never be utilized which is why they created ETH to replace it with is trading above $100 now. Compare that with DOGE which is being utilized more and more each day by large corporations and companies as an actual currency. I believe DOGE is still at a discount right now.	                           
     .  200615614713    .    <A|950181551> Today was a crazy day hahha but $HH$Let this run its course$HH$Let it rise and let it fall $HH$Let it climb to 80 and dip to 40 $HH$This will dip regardless and no one knows when $HH$But dont make your future self regret selling because it dropped 10 cents in a few seconds$HH$$HH$Plus, if it drops hard.. Just buy more $HH$once this is 5-10 youll be really happy you stopped selling $HH$I bought at .03 and sold at 0.07 and doing that is my current regret $HH$$HH$$HH$Dont worry though i bought back in at .08 currently up 600 buckaroos 
