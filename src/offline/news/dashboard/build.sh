
echo "------------------------------------------------ "
Stage=$1
if [[ -z $Stage ]];then
  Stage='dev'
fi

echo "Stage=$Stage"

repoName=rs/news-dashboard

../spark_build.sh $repoName $Stage

