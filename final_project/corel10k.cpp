#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <limits>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#define DIMENSIONS 256
#define N 10000
#define DEBUG 1

using namespace std;
using namespace cv;

std::vector<unsigned char> LBPDescriptor[N+1];

/// Default parameters
int LBPDimensions   = 256;
int FDimensions     = 40;

int numInstances    = 9500;
int numDimensions   = 256;
int K               = 5;

string datasetPath  = "../converted_images/";
string suffix       = ".jpg";

string LBPDescFile  = "lbpDescriptors1.txt";
string FDescFile    = "fourierDescriptors1.txt";
string DescFileName = "";
string MapFileName  = "indexImg.map.txt";

template <class T>
void lbp(const Mat &src, Mat &dst) {
    unsigned char lbpCode=0;
    dst = Mat::zeros(src.rows-2, src.cols-2, CV_8UC1);
    for (int i=1; i<src.rows-1; i++) {
        for (int j=1; j<src.cols-1; j++) {
            T center = src.at<T>(i,j);
            //top-left
            lbpCode = (src.at<T>(i-1, j-1) > center) << 7;
            //top
            lbpCode |= (src.at<T>(i-1, j  ) > center) << 6;
            //top-right
            lbpCode |= (src.at<T>(i-1, j+1) > center) << 5;
            //right
            lbpCode |= (src.at<T>(i  , j+1) > center) << 4;
            //bottom-right
            lbpCode |= (src.at<T>(i+1, j+1) > center) << 3;
            //bottom
            lbpCode |= (src.at<T>(i+1, j  ) > center) << 2;
            //bottom-left
            lbpCode |= (src.at<T>(i+1, j-1) > center) << 1;
            //left
            lbpCode |= (src.at<T>(i  , j-1) > center) << 0;

            dst.at<unsigned char>(i-1, j-1) = lbpCode;
        }
    }
}

// Extended Local Binery pattern
template <typename _Tp>
void elbp(const Mat& src, Mat& dst, int radius, int neighbors=8) {
    neighbors = max(min(neighbors,31),1); // set bounds..
    dst = Mat::zeros(src.rows-2*radius, src.cols-2*radius, CV_32SC1);
    for(int n=0; n<neighbors; n++) {
        // sample points
        float x = static_cast<float>(radius) * cos(2.0*M_PI*n/static_cast<float>(neighbors));
        float y = static_cast<float>(radius) * -sin(2.0*M_PI*n/static_cast<float>(neighbors));
        // relative indices
        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));
        // fractional part
        float ty = y - fy;
        float tx = x - fx;
        // set interpolation weights
        float w1 = (1 - tx) * (1 - ty);
        float w2 =      tx  * (1 - ty);
        float w3 = (1 - tx) *      ty;
        float w4 =      tx  *      ty;
        // iterate through your data
        for(int i=radius; i < src.rows-radius;i++) {
            for(int j=radius;j < src.cols-radius;j++) {
                float t = w1*src.at<_Tp>(i+fy,j+fx) + w2*src.at<_Tp>(i+fy,j+cx) + w3*src.at<_Tp>(i+cy,j+fx) + w4*src.at<_Tp>(i+cy,j+cx);
                // we are dealing with floating point precision, so add some little tolerance
                dst.at<unsigned int>(i-radius,j-radius) += ((t > src.at<_Tp>(i,j)) && (abs(t-src.at<_Tp>(i,j)) > std::numeric_limits<float>::epsilon())) << n;
            }
        }
    }
}

template <class T>
std::vector<T> getHistogram(const Mat &img) {
    std::vector<T> h(DIMENSIONS, 0);
    for (int i=0; i<img.rows; i++) {
        for (int j=0; j<img.cols; j++) {
            h[img.at<T>(i,j)]++;
        }
    }
    return h;
}

template <typename T>
void printHistogram(const std::vector<T> h) {
    for (int i=0; i<DIMENSIONS; i++) {
        cout << "[" << i << "]: " << (int)h[i] << "\n";
    }
}

/**
 * Computes the similarity of two image descriptors
 * @param queryDescriptor       the query image descriptor
 * @param anotherDescriptor     the other image descriptor
 * @return the negative euclidian distance between the descriptors
 */
template <typename T>
double similarity(const std::vector<T> &queryDescriptor, const std::vector<T> &anotherDescriptor) {
    double acc=0.0;
    if (queryDescriptor.size() != anotherDescriptor.size()) {
        cerr << "ERROR: queryDescriptor and anotherDescriptor have different dimensions.\n";
        return -1;
    }
    for (int i=0; i<queryDescriptor.size(); i++) {
        acc += (queryDescriptor[i] - anotherDescriptor[i]) * (queryDescriptor[i] - anotherDescriptor[i]);
    }
    return sqrt(acc);
}


vector<string> &split(const string &s, char delim, vector<string> &elems) {
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

vector<string> split(const string &s, char delim) {
    vector<string> elems;
    split(s, delim, elems);
    return elems;
}

void parseInput(int start, int argc, char **argv) {
    for(int i = start ; i < argc ; i+=2){
        if(!strcmp(argv[i], "-numInstances")){ // Help message
            numInstances = atoi(argv[i+1]);
            if (DEBUG) cerr << "numInstances: " << numInstances <<endl;
        }
        else if (!strcmp(argv[i], "-numDimensions")) {
            numDimensions = atoi(argv[i+1]);
            if (DEBUG) cerr << "numDimensions: " << numDimensions <<endl;
        }
        else if (!strcmp(argv[i], "-d")) { // descriptors file
            DescFileName = argv[i+1];
            if (DEBUG) cerr << "DescFileName: " <<DescFileName<<endl;
        }
    }
}

int main(int argc, char** argv)
{
    // Constants
    string ERROR_MSG_USAGE = "usage: corel10k <imagem de consulta> <k-próximos> <método> <params>\n";
    //-----------------------------------------------------------------------------------------------
    
    ifstream inFile;
    Mat queryImage, lbpQueryImage;
    std::vector<unsigned char> queryDescriptor;
    vector<string> imageFileNames;

    /// Generate LBP descriptors ./corel10k -glbp <path> -r <radius>
    if ( argc > 1 && !strcmp(argv[1], "-glbp") ) {
        ofstream outFile, mapFile;
        string path;
        string imgPath;
        string outFileName;
        int radius;

        /// get the radius 
        radius = (argc >= 5 && !strcmp(argv[3], "-r")) ? atoi(argv[4]) : 1;
        cout << "radius: " <<radius<<endl;
        stringstream aux; aux << radius;
        LBPDescFile = "lbpDescriptors_r"+aux.str()+".txt";
        cout << "LBPDescFile: " <<LBPDescFile<<endl;

        outFile.open(LBPDescFile.c_str(), std::ofstream::out);
        mapFile.open(MapFileName.c_str(), std::ofstream::out);
            
        if (!outFile.is_open()) {
            cerr << "ERROR: could not open the file " << LBPDescFile;
            return -1;
        }

        path = argv[2];

        /// Generate the descriptors to all the images in the path
        /// Iterate through the images and generate the correspondent descriptors
        string descStr="", mapStr="";
        int idxCnt=0;
        cout << "Processing ... " << setw(5) << setfill(' ') << "\%"; cout << "\b\b\b\b\b\b";
        cout.flush();
        
        for (int i=1; i<=N; i++) {
            if (i%10 == 0) {
                cout  << setw(5) << 100*i/(float)N; 
                if ( i < N ) { 
                    cout << "\b\b\b\b\b";
                }
                else
                    cout << "\n\n";
            }

            stringstream ssImgNumber;
            ssImgNumber << i;
            imgPath = path+"/"+ssImgNumber.str()+".jpg";

            Mat img = imread(imgPath, IMREAD_GRAYSCALE);
            if (img.data) { // If img exists
                Mat lbpImage;
                // lbp<unsigned char>(img, lbpImage);
                elbp<unsigned char>(img, lbpImage, radius);
                LBPDescriptor[i] = getHistogram<unsigned char>(lbpImage);
                
                /// Convert the descriptors to string and stores into descStr
                for (int j=0; j<LBPDescriptor[i].size(); j++) {
                    stringstream ss2;
                    ss2 << (int)LBPDescriptor[i][j];
                    descStr += ss2.str()+" ";
                }
                descStr += "\n";

                stringstream ssIndex; ssIndex << idxCnt++;
                mapStr += ssIndex.str()+" "+ssImgNumber.str()+"\n";
            }
        }
        mapFile << mapStr;
        outFile << descStr;
        mapFile.close();
        outFile.close();

        cout << "DONE!\n=> Two files generated: " << LBPDescFile << " and " << MapFileName << "\n\n";
    }
    /// Search k most similar images
    else if ( argc >= 4 ) {
        ifstream inMapFile;
        map<int, int> mapIdxImg;
        map<int, int> mapImgIdx;
        // ./corel10k <imagem de consulta> <k-próximos> <método> [parâmetros adicionais do método]
        // ./corel10k 200.jpg 5 lbp -d lbpDescriptors1.txt -numInstances 9500 -numDimensions 256

        /// Get the number of neighbors to be considered
        /// Add 1 because the query image is in the dataset
        K = atoi(argv[2])+1;

        /// Set default parameters
        /// Local Binary Pattern
        if (!strcmp(argv[3], "lbp")) {
            DescFileName = LBPDescFile;
            numDimensions= LBPDimensions;
        }
        else if (!strcmp(argv[3], "fourier")) {
            DescFileName = FDescFile;
            numDimensions= FDimensions;
        }
        else {
            cerr << ERROR_MSG_USAGE;
            return -1;
        }

        /// Parse the input PARAMS only
        parseInput(4, argc, argv);

        /// Load the map file
        int key, value;
        inMapFile.open(MapFileName.c_str(), std::ifstream::in);
        if (!inMapFile.is_open()) {
            cerr << "ERROR: could  not find the map file. Please put the map file in the same folder of the corel10k executable.\n";
            return -1;
        }
        while (inMapFile >> key >> value) {
            mapIdxImg[key]      = value;
            mapImgIdx[value]    = key;
        }

        /// Get the index of the query image
        int queryImageIndex, queryImageNumber;
        queryImageNumber = atoi((split(argv[1], '.')[0].c_str()));
        if (mapImgIdx.count(queryImageNumber) == 0) {
            cerr << "This image is corrupted! Please, don't choose an image in the range [1001 1500].\n";
            return -1;
        }
        queryImageIndex = mapImgIdx[queryImageNumber];

        Mat features;
        int freq;
        vector<float> queryImageDescriptor(numDimensions);
        vector<int> index(K);
        vector<float> dist(K);

        inFile.open(DescFileName.c_str(), std::ifstream::in);
        if (inFile.is_open()) {
            // void Mat::create(int rows, int cols, int type)
            features.create(numInstances, numDimensions, CV_32F);
            for (int i=0; i<features.rows; i++) {
                for (int j=0; j<features.cols; j++) {
                    inFile >> freq;
                    features.at<float>(i,j) = (float)freq;
                    // cout << features.at<float>(i,j) << " ";
                }
                // cout << endl;
            }

            /// Load query image descriptor
            for (int j=0; j<features.cols; j++) {
                queryImageDescriptor[j] = features.at<float>(queryImageIndex, j);
            }

            /// Set the number of KDTrees
            flann::KDTreeIndexParams indexParams(5);
            
            /// Create index
            flann::Index kdTree(features, indexParams);
            
            /// Search
            kdTree.knnSearch(queryImageDescriptor, index, dist, K, flann::SearchParams(128));

            /// Print the k most similar images to the query image
            for (int i=1; i<index.size(); i++) {
                cout << mapIdxImg[ index[i] ] << "\n";
                // cout << "index: " << index[i] << ", dist: " << dist[i]<<endl;
            }
            inFile.close();
        }
    } 
    else {
        cerr << ERROR_MSG_USAGE;
        return -1;
    }


    return 0;
}