import Augmentor
p=Augmentor.Pipeline("bacterial leaf spot/")
p.rotate(probability=0.5, max_left_rotation=20,max_right_rotation=25)
p.zoom(probability=0.2,min_factor=1.0,max_factor=1.5)
p.random_distortion(probability=1, grid_width=3,grid_height=3,magnitude=7)
p.flip_left_right(probability=0.7) 
p.rotate_without_crop(probability=0.5, max_left_rotation=10, max_right_rotation=20, expand=False)
p.flip_top_bottom(probability=0.3)
p.sample(100)
p.process
