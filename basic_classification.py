from sklearn.neighbors import KNeighborsRegressor
import numpy as np

class BasicClassification:
    def _init_(self, n_neighbors=3):  
        self.n_neighbors = n_neighbors
        self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors)
    
    def fit(self, train_data, train_labels):
        self.model.fit(np.array(train_data).reshape(-1, 1), train_labels)
    
    def predict(self, sample):
        return self.model.predict(np.array(sample).reshape(-1, 1))

train_data = [4, 6, 8, 4, 6, 8, 8] 
train_labels = [20000, 25000, 30000, 21000, 26000, 31000, 29500] 

knn_classifier = BasicClassification(n_neighbors=3)
knn_classifier.fit(train_data, train_labels)

silindir_sayisi = int(input("Tahmin etmek istediğiniz motorun silindir sayısını girin: "))

predicted_price = knn_classifier.predict([silindir_sayisi])
print(f"{silindir_sayisi} silindirli motor için tahmin edilen fiyat: {predicted_price[0]:.2f}")