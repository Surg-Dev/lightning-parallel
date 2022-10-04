docker build -t lumosquad .
img_id=$(docker create lumosquad)
docker cp $img_id:/usr/src/lumosquad/lumosquad .