import builtins
import numpy as np

class KNearestNeighbor(object):
    """ a kNN classifier with Cosine and L2 distances """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0, distfn='L2'):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            if distfn == 'L2':
                dists = self.compute_L2_distances_no_loops(X)
            else:
                dists = self.compute_Cosine_distances_no_loops(X)
        elif num_loops == 1:
            if distfn == 'L2':
                dists = self.compute_L2_distances_one_loop(X)
            else:
                dists = self.compute_Cosine_distances_one_loop(X)
        elif num_loops == 2:
            if distfn == 'L2':
                dists = self.compute_L2_distances_two_loops(X)
            else:
                dists = self.compute_Cosine_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_L2_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

                # d(i,j) = euclidian distance between Xi and Yi
                dists[i, j] = np.sqrt(np.sum((X[i] - self.X_train[j])**2))
                
                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_L2_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_L2_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            
            squared_diff = (X[i] - self.X_train) ** 2 # computing L2 distance for Xi on every training data
            sum_squared_diff = np.sum(squared_diff, axis=1) # np.sum() computes sum of matrixes
            dists[i] = np.sqrt(sum_squared_diff) # Filling in the distance matrix 
            
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_L2_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_L2_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        test_sq = np.sum(X**2, axis=1, keepdims=True) # a^2
        train_sq = np.sum(self.X_train**2, axis=1) # b^2

        # we have a^2 and b^2, now we need 2*a*b. We can find this intermediate matrix by performing dot product on b's transpose
        
        intermediate = np.dot(X, self.X_train.T)

        # Perform -2*a*b and compute distance

        dists = np.sqrt(test_sq + train_sq - 2 * intermediate)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_Cosine_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Cosine distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                              #
                # Compute the cosine distance between the ith test point and the jth #
                # training point, and store the result in dists[i, j]. You should    #
                # not use a loop over dimension, nor use np.linalg.norm() and
                # scipy.spatial.distance.cosine
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                
                # we will compute 1 - ab / sq(a^2) + sq(b^2)

                dot_product = np.dot(X[i], self.X_train[j]) # ab
                sq2_test = np.sqrt(np.sum(X[i]**2)) # abs(A)
                sq2_train = np.sqrt(np.sum(self.X_train[j]**2)) # abs(B)

                cosine_distance = 1 - (dot_product / (sq2_test + sq2_train))

                dists[i, j] = cosine_distance

                
                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_Cosine_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_Cosine_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the cosine distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm(). and scipy.spatial.distance.cosine      #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            
            dot_product = np.dot(X[i], self.X_train.T) 

            sq2_test = np.sqrt(np.sum(X[i]**2))
            sq2_train = np.sqrt(np.sum(self.X_train**2, axis=1))

            cosine_distance = 1 - (dot_product / (sq2_test + sq2_train))

            dists[i] = cosine_distance
            
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists
    

    def compute_Cosine_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_Cosine_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the cosine distance between all test points and all training  #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm() or scipy.spatial.distance.cosine             #
        #                                                                       #
        #                                                                       #
        #                                                                       #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        dot_product = np.dot(X, self.X_train.T)

        sq2_test = np.sqrt(np.sum(X**2, axis=1))
        sq2_train = np.sqrt(np.sum(self.X_train**2, axis=1))

        cosine_distance = 1 - (dot_product / (sq2_test + sq2_train))

        dists = cosine_distance


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            sorted_indices = np.argsort(dists[i]) # sort indices in dists(i)
            k_nearest_indices = sorted_indices[:k] # first k indices

            for j in k_nearest_indices: # add k nearest neigbors to closest y
                closest_y.append(self.y_train[j])



            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            label_counts = np.bincount(closest_y) # counts how many times each unique label appears

            most_common_label = np.argmax(label_counts) # finds most common argument

            y_pred[i] = most_common_label

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
