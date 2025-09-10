# VU Recommender Systems - Exercises with rrecsys
# =================================================

# 1. Install rrecsys from GitHub and load the library
# Note: You may need to install devtools first if not already installed
# install.packages("devtools")

# Install rrecsys from GitHub
# if (!require(rrecsys)) {
#   devtools::install_github("ludovikcoba/rrecsys")
# }

# Load required libraries
library(rrecsys)
library(ggplot2)

print("Step 1: rrecsys library loaded successfully")

#Load data - MovieLens 
data(mlLatest100k)

# View dataset 
View(mlLatest100k)

# show structure of the dataset
print(paste("Dataset dimensions:", nrow(mlLatest100k), "x", ncol(mlLatest100k)))

# define data in rrecsys format
ml_data <- defineData(mlLatest100k, minimum = .1, maximum = 5, intScale = TRUE)

# show summary of the dataset
summary(ml_data)



# Number of times an item was rated.
number_of_ratings_per_item <- colRatings(ml_data)
print(head(number_of_ratings_per_item))

# Number of times a user has rated.
number_of_ratings_per_users <- rowRatings(ml_data)
print(head(number_of_ratings_per_users, 10))

# Total number of rating in the rating matrix.
total_number_of_rating <- numRatings(ml_data)
print(paste("Total number of ratings:", total_number_of_rating))

# Sparsity.
sparsity_value <- sparsity(ml_data)
print(paste("Sparsity:", round(sparsity_value*100, 2), "%"))


#plot histogram of ratings per item
# ggplot(data.frame(num_ratings = number_of_ratings_per_item), aes(x = num_ratings)) +
#   geom_histogram(binwidth = 1, fill = "blue", color = "black", alpha = 0.7) +
#   labs(title = "Histogram of Number of Ratings per Item",
#        x = "Number of Ratings",
#        y = "Count of Items") +
#   theme_minimal()
print(histogram(ml_data))


# Plot long tail distribution of ratings per item

sorted_item_ratings <- sort(number_of_ratings_per_item, decreasing = TRUE)


plot_data <- data.frame(
  item_rank = 1:length(sorted_item_ratings),
  num_ratings = sorted_item_ratings
)

# # Create the long tail plot
longtail_plot <- ggplot(plot_data, aes(x = item_rank, y = num_ratings)) +
  geom_line(color = "blue", size = 1) +
  geom_point(color = "red", alpha = 0.6, size = 0.5) +
  
  labs(title = "Long Tail Distribution: Number of Ratings per Item",
       x = "Item Rank",
       y = "Number of Ratings",
       caption = "Items sorted by number of ratings (descending)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

print(longtail_plot)

# make dataset have sparsity of <=80%
smallMl <- ml_data[rowRatings(ml_data)>=50, colRatings(ml_data)>=50]
print(paste("New sparsity:", round(sparsity(smallMl)*100, 2), "%"))

# binarize ml_data
binML <- defineData(mlLatest100k, positiveThreshold = 3, binary=TRUE)
print(paste("Binarized sparsity:", round(sparsity(binML)*100, 2), "%"))
binML <- binML[rowRatings(binML)>=50, colRatings(binML)>=50]
print(paste("New Binarized sparsity:", round(sparsity(binML)*100, 2), "%"))


# Alice experiments

a <- c(5, 3, 4, 4, 0, 3, 1, 2, 3, 3, 4, 3, 4, 3, 5, 3, 3, 1, 5,4, 1, 5, 5, 2, 1)
d <- matrix(a, nrow = 5, byrow = T)
rownames(d) <- paste0("user", 1:5)
colnames(d) <- paste0("item", 1:5)
rownames(d)[1] <- "Alice"

d[1,5] <- NA

View(d)


# Prepare data and train the item-based model
data_set <- defineData(d, minimum = 1, maximum = 5)
ibcf_model <- rrecsys(data_set, alg = "IBKNN", neigh = 2, simFunct = "cos")

# Print similarity matrix
print("Similarity Matrix:")
print(ibcf_model@sim)

# Print most similar items to item5
print("Most similar items to item5:")
print(ibcf_model@sim_index_kNN[5, ])


# predict
ibcf_predictions <- predict(ibcf_model)
# ibcf_predictions["Alice", "item5"]
print(ibcf_predictions)

# Train SVD Model (2 Features) and Predict
svd_model <- rrecsys(data_set, alg = "FunkSVD", k = 2)
svd_predictions <- predict(svd_model)
# svd_predictions["Alice", "item5"]
print(svd_predictions)

