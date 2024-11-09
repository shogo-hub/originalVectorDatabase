#file  locatio is disided the relative path from exe file 
import sqlite3
import numpy as np 

class DatabaseManagement:
    def __init__(self,dbName):
        self.connection = sqlite3.connect("textAttributedGraph.db")
        self.dbName = dbName
    #Several function 
    #About location of database, I will decide later.
    def createCursor(self)->sqlite3.Connection.cursor:
        return self.connection.cursor()
    
    
    
    


    #<create table method>
    def createTable(self,tableName):
        cursor = self.createCursor()
        cursor.execute(f"""
            CREATE TABLE IF NOT {tableName}
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT
                embedding_vector Text
        """)
        self.connection.commit()
    
    
    #crud function 
    def getColumnData(self, columnName1="id", columnName2="embeddingVector", isEmbeddingVectorRetrieving=True) -> np.ndarray:
        """
        This method is responsible for retrieving all data from two specific columns 
        in the table.
        """
        cursor = self.createCursor()

        # Retrieve all data from the specified columns
        cursor.execute(f"SELECT {columnName1}, {columnName2} FROM my_table")
        results = cursor.fetchall()  # Fetch all rows

        if results:
            if isEmbeddingVectorRetrieving:
                # Convert each embedding vector string to a NumPy array
                data = [[row[0], self.convertStringIntoMatrix(row[1])] for row in results]
            else:
                # Only retrieve the first column as a NumPy array
                data = [ [row[0],row[1] ] for row in results]
            
            # Convert the list of tuples to a NumPy array for consistency
            return np.array(data, dtype=object)
        else:
            return np.array([])

    def convertStringIntoMatrix(self, embeddingString: str) -> np.ndarray:
        """
        Convert a string of comma-separated values into a NumPy array.
        For example: "1,2,3,4" -> np.array([1, 2, 3, 4])
        """
        # Split the string by commas and convert it into a list of floats or integers
        embedding_list = embeddingString.split(',')
        
        # Convert the list into a NumPy array of floats (or integers, depending on the data)
        embedding_array = np.array([float(i) for i in embedding_list])

        return embedding_array

    def addRowData(self, id, columnName, value):
        """
        This method updates a specific column in a row identified by `id` in the SQLite database.
        
        Parameters:
        - id: The unique identifier of the row to update.
        - columnName: The name of the column to be updated.
        - value: The value to set for the specified column.
        """
        cursor = self.createCursor()
        
        # Use a parameterized query to avoid SQL injection
        query = f"UPDATE my_table SET {columnName} = ? WHERE id = ?"
        
        try:
            # Execute the update statement with the provided value and id
            cursor.execute(query, (value, id))
            self.connection.commit()
            print(f"Row with id {id} successfully updated in column {columnName}.")
            
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")

        