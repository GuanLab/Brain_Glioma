#!/usr/bin/perl
#

# Script for evenly spliting the dataset into training set and testing set.

if (scalar (@ARGV)<1){ print "perl XXX 0 (to 9)\n";
	die;
}
srand($ARGV[0]);
#record train individual and testing individual
#
@all_pid=glob "/data/HGG_np_flair/*";
open TRAIN, ">train_id.txt" or die;
open TEST, ">test_id.txt" or die;

foreach $dir (@all_pid){
	@t=split '/', $dir;
	$id=$t[-1];
	$r=rand(1);
	if ($r<0.5){
		$train{$id}=0;
		print TRAIN "$id\n";
	}else{
		$test{$id}=0;
		print TEST "$id\n";
	}
}


close TRAIN;
close TEST;

# original images from axial view
@mat=glob "/data/HGG_np_flair/*/*";
open NEW, ">train_gs.dat" or die;
open TEST, ">test_gs.dat" or die;
foreach $file (@mat){
 	@t=split '/', $file;
	$id=pop @t;
	$id=pop @t;
	if (exists $train{$id}){
		print NEW "$file";
		$new_file=$file;
		$new_file=~s/HGG_np_flair/HGG_np_seg/g;
		if (-f $new_file){
			print NEW "\t$new_file\n";
		}else{
			print NEW "\tneg\n";
		}
	}else{
		print TEST "$file";
		$new_file=$file;
		$new_file=~s/HGG_np_flair/HGG_np_seg/g;
		if (-f $new_file){
			print TEST "\t$new_file\n";
		}else{
			print TEST "\tneg\n";
		}
	}

}


# /data_1st_asslide contains images from sagittal view
@mat=glob "/data_1st_asslide/HGG_np_flair/*/*";
foreach $file (@mat){
 	@t=split '/', $file;
	$id=pop @t;
	$id=pop @t;
	if (exists $train{$id}){
		print NEW "$file";
		$new_file=$file;
		$new_file=~s/HGG_np_flair/HGG_np_seg/g;
		if (-f $new_file){
			print NEW "\t$new_file\n";
		}else{
			print NEW "\tneg\n";
		}
	}else{
	}

}


# /data_2nd_asslide contains images from coronal view
@mat=glob "/data_2nd_asslide/HGG_np_flair/*/*";
foreach $file (@mat){
 	@t=split '/', $file;
	$id=pop @t;
	$id=pop @t;
	if (exists $train{$id}){
		print NEW "$file";
		$new_file=$file;
		$new_file=~s/HGG_np_flair/HGG_np_seg/g;
		if (-f $new_file){
			print NEW "\t$new_file\n";
		}else{
			print NEW "\tneg\n";
		}
	}else{
	}

}


# original images from axial view
@mat=glob "/data/LGG_np_flair/*/*";
foreach $file (@mat){
 	@t=split '/', $file;
	$id=pop @t;
	$id=pop @t;
		print NEW "$file";
		$new_file=$file;
		$new_file=~s/LGG_np_flair/LGG_np_seg/g;
		if (-f $new_file){
			print NEW "\t$new_file\n";
		}else{
			print NEW "\tneg\n";
		}

}


# /data_1st_asslide contains images from sagittal view
@mat=glob "/data_1st_asslide/LGG_np_flair/*/*";
foreach $file (@mat){
 	@t=split '/', $file;
	$id=pop @t;
	$id=pop @t;
		print NEW "$file";
		$new_file=$file;
		$new_file=~s/LGG_np_flair/LGG_np_seg/g;
		if (-f $new_file){
			print NEW "\t$new_file\n";
		}else{
			print NEW "\tneg\n";
		}

}


# /data_2nd_asslide contains images from coronal view
@mat=glob "/data_2nd_asslide/LGG_np_flair/*/*";
foreach $file (@mat){
 	@t=split '/', $file;
	$id=pop @t;
	$id=pop @t;
		print NEW "$file";
		$new_file=$file;
		$new_file=~s/LGG_np_flair/LGG_np_seg/g;
		if (-f $new_file){
			print NEW "\t$new_file\n";
		}else{
			print NEW "\tneg\n";
		}
}
close NEW;


