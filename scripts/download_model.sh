if [ ! -d "./output" ]; then
	mkdir output
fi

wget https://s3.eu-central-1.amazonaws.com/avg-projects/convolutional_occupancy_networks/models/pointcloud_crop/room_grid64.pt -O ./output/room_grid64.pt
