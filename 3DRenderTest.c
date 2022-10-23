#include <stdio.h>
#include <math.h>
#include <float.h>

#define MAXTRIS 256
#define HEIGHT 50
#define WIDTH 50
#define FOV 10
#define DELAY 10000000
#define GRAYSCALE " .:-=+*#%@"
#define MAXNAMELEN 64

enum Filetype { OBJ , TXT , PLY , STL };

/* FILE FORMAT (for .txt files):
*  1 triangle {
*  x, y, z
*  x, y, z
*  x, y, z
*  
*  }
*   triangles are separated by a blank line, doesn't matter to the program but makes it easier to read
*   points should be listed counterclockwise in order to calculate normals correctly (right hand rule)
*/

typedef struct {
    double x;
    double y;
    double z;
} Point;

typedef struct {
    double I;
    double J;
    double K;
} Vector;

typedef struct {
    Point verts[3];
    Vector normal;
    double D;
} Triangle;

typedef struct {
    Triangle tris[MAXTRIS];
    int triCount;
} Mesh;

typedef struct {
    double x;
    double y;
    double z;
    double I;
    double J;
    double K;
    double FieldOfView;
    int outputHeight;
    int outputWidth;
} Camera;

typedef struct {
    double x;
    double y;
    double z;
    double intensity;
} Light;

void wait(double delay);
void enterBuffer(void);
void clearScreen(void);
void buildMesh(Mesh* mesh, FILE* fp, const enum Filetype* type);
void display(const Mesh* mesh, const Camera* camera, const Light lights[]);
void visualizeCameraAngle(const Camera* camera, const Mesh* mesh);
void orbitCamera(Camera* camera, const Point* center, const double thetaChange, const double phiChange);
void vectorToAngle(Vector* v, double* thetaChange, double* phiChange);
void rotateVector(Vector* v, const double thetaChange, const double phiChange);
void swap(int indexA, int indexB, double arr[]);
void quickSort(int left, int right, double arr[]);
void flipArray(double arr[], int len);
int checkWithinTriangle(const Point point, const Triangle tri);
int partition(int left, int right, double pivotValue, double arr[]);
double calculateMagnitude(Vector* v);
double degToRad(double deg);
double radToDeg(double rad);
double round(double num, int precision);
double dotProduct(const Vector a, const Vector b);
Vector angleToVector(const double thetaChange, const double phiChange);
Vector calculateNormal(const Triangle* tri);
Vector crossProduct(const Vector* rAB, const Vector* rBC);
Light initializeLight(const double x, const double y, const double z, const double intensity);
Point getCenterOfMesh(const Mesh* mesh);
void calculateIntersectDistances(const Mesh* mesh, const Point origin, const Vector* rays, const int rayCount, double intersectDistances[]);
FILE* getUserFile(enum Filetype* type);

int main()
{
    enum Filetype type;
    FILE* fp = getUserFile(&type);
    Mesh mesh;
    buildMesh(&mesh, fp, &type);
    Point center = getCenterOfMesh(&mesh);
    Camera camera = {0, 5, -5, 0, -5 / sqrt(50), 5 / sqrt(50), FOV, HEIGHT, WIDTH};
    Light lights[2];
    lights[0] = initializeLight(5, 5, -5, 1);
    lights[1] = initializeLight(-5, -5, -5, 0.25);
    
    while (1) {
        clearScreen();
        display(&mesh, &camera, lights);

        orbitCamera(&camera, &center, 0.5, 0);
        wait(DELAY);

        enterBuffer();
    }
    
    return 0;
}

void swap(int indexA, int indexB, double arr[])
{
    double temp = arr[indexA];
    arr[indexA] = arr[indexB];
    arr[indexB] = temp;
}

int partition(int left, int right, double pivotValue, double arr[])
{
    int leftIndex = left - 1;
    int rightIndex = right;

    while (leftIndex < rightIndex) {
        while (arr[++leftIndex] < pivotValue);
        while (rightIndex > 0 && arr[--rightIndex] > pivotValue);

        if (leftIndex < rightIndex) {
            swap(leftIndex, rightIndex, arr);
        }
    }
    swap(leftIndex, right, arr);

    return leftIndex;
}

void quickSort(int left, int right, double arr[])
{
    if (right - left > 0) {
        double pivot = arr[right];
        int partitionPoint = partition(left, right, pivot, arr);
        quickSort(left, partitionPoint - 1, arr);
        quickSort(partitionPoint + 1, right, arr);
    }
}

void flipArray(double arr[], int len)
{
    int i = 0;
    int j = len - 1;

    for (i = 0, j = len - 1; i < len / 2; ++i, --j)
        swap(i, j, arr);
}

void wait(double delay)
{
    int i = 0;
    float value = 2;
    
    for (i = 0; i < delay; ++i)
        value = 1 / value;
}

void enterBuffer(void)
{
    char garbageInput = 'a';

    while (scanf("%c", &garbageInput) && garbageInput != '\n');
}

void clearScreen(void)
{
    printf("\033c");
}

void buildMesh(Mesh* mesh, FILE* fp, const enum Filetype* type)
{
    int vert = 0;
    int tri = 0;
    int i = 0;
    double input = 0;
    Vector vertVec = {};
    
    switch (*type) {
        case TXT:
            while (fscanf(fp, "%lf", &input) > 0 && tri < MAXTRIS) {
                if (i > 2) {
                    i = 0;
                    vert++;
                }
                if (vert > 2) {
                    vert = 0;
                    tri++;
                }
                switch (i) {
                    case 0:
                        mesh->tris[tri].verts[vert].x = input;
                        break;
                    case 1:
                        mesh->tris[tri].verts[vert].y = input;
                        break;
                    case 2:
                        mesh->tris[tri].verts[vert].z = input;
                        break;
                }
                ++i;
            }
            mesh->triCount = i + 1;
            for (i = 0; i < mesh->triCount; ++i) {
                vertVec.I = mesh->tris[i].verts[0].x;
                vertVec.J = mesh->tris[i].verts[0].y;
                vertVec.K = mesh->tris[i].verts[0].z;
                mesh->tris[i].normal = calculateNormal(&mesh->tris[i]);
                mesh->tris[i].D = dotProduct(mesh->tris[i].normal, vertVec);
            }
            break;
        case OBJ:
            
            break;
        case PLY:
            
            break;
        case STL:
            
            break;
    }
}

void display(const Mesh* mesh, const Camera* camera, const Light lights[])
{
    int i = 0;
    int j = 0;
    int k = 0;
    int rayCount = WIDTH;
    int rowCount = HEIGHT;
    int triCount = mesh->triCount;
    double rayAngleDiff = ((double)(FOV)) / rayCount;
    Vector camDir = {camera->I, camera->J, camera->K};
    double theta = 0;
    double phi = 0;
    Vector rays[rayCount];
    double intersectDistances[rayCount * triCount];
    Point origin = {camera->x, camera->y, camera->z};
    vectorToAngle(&camDir, &theta, &phi);
    double* rows[rowCount];
    int printChar = 0;

    phi -= (rayAngleDiff * rowCount) / 2.0;
    for (j = 0; j < rowCount; ++j) {
        theta -= (rayAngleDiff * rayCount) / 2.0;
        for (i = 0; i < rayCount; ++i) {
            rays[i] = angleToVector(theta, phi);
            theta += rayAngleDiff;
        }

        calculateIntersectDistances(mesh, origin, rays, rayCount, intersectDistances);

        rows[j] = intersectDistances;
        phi += rayAngleDiff;
    }

    for (i = 0; i < rowCount; ++i) {
        for (j = 0; j < rayCount; ++j) {
            printChar = 0;
            for (k = 0; k < triCount; ++k) {
                if (rows[i][(rayCount - 1 - j) + (triCount - 1 - k)] > 0)
                    printChar = 1;
            }
            if (printChar == 1 && i == 180 && j == 1)
                fputc('1', stdout);
            else if (printChar == 0 && i == 180 && j == 1)
                fputc('0', stdout);
            else if (printChar == 1)
                fputc('@', stdout);
            else
                fputc(' ', stdout);
        }
        fputc('\n', stdout);
    }

    vectorToAngle(&camDir, &theta, &phi);
    printf("Camera angle: theta = %lf, phi = %lf\n", theta, phi);
    printf("Camera position: x = %lf, y = %lf, z = %lf\n", origin.x, origin.y, origin.z);
    printf("Ray angle difference: %lf, Ray count: %d, total theta covered: %lf, total phi covered: %lf\n", rayAngleDiff, rayCount, rayCount * rayAngleDiff, rowCount * rayAngleDiff);
    printf("Pixel (%d, %d), has a distance of %lf\n", 1, 40, rows[40][1]);

    /*
    Triangle tri;
    printf("There are %d triangles in the mesh\n", mesh->triCount);
    for (i = 0; i < mesh->triCount; ++i) {
        tri = mesh->tris[i];
        printf("Triangle %d normal: I = %lf, J = %lf, K = %lf\n", i, tri.normal.I, tri.normal.J, tri.normal.K);
        for (j = 0; j < 3; j++)
            printf("Triangle %d point %d: x = %lf, y = %lf, z = %lf\n", i, j, tri.verts[j].x, tri.verts[j].y, tri.verts[j].z);
        printf("\n");
    }
    
    printf("Camera: x = %lf, y = %lf, z = %lf; I = %lf, J = %lf, K = %lf\n", camera->x, camera->y, camera->z, camera->I, camera->J, camera->K); */

    /* visualizeCameraAngle(camera, mesh); */
}

void calculateIntersectDistances(const Mesh* mesh, const Point origin, const Vector* rays, const int rayCount, double intersectDistances[])
{
    int i = 0;
    int j = 0;
    Vector normal = {};
    Vector ray = {};
    Vector originVector = {origin.x, origin.y, origin.z};
    Point intersection = {};
    double intersectDistance = 0;

    for (i = 0; i < rayCount; ++i) {
        for (j = 0; j < mesh->triCount; ++j) {
            normal = mesh->tris[j].normal;
            ray = rays[i];

            if (dotProduct(normal, ray) == 0.0) {
                intersectDistances[i + j] = -1;
            } else {
                intersectDistance = (mesh->tris[j].D - dotProduct(normal, originVector)) / dotProduct(normal, ray);
                intersectDistance = (intersectDistance < 0) ? -intersectDistance : intersectDistance;

                intersection.x = ray.I * intersectDistance + origin.x;
                intersection.y = ray.J * intersectDistance + origin.y;
                intersection.z = ray.K * intersectDistance + origin.z;

                intersectDistances[i + j] = checkWithinTriangle(intersection, mesh->tris[j]) ? intersectDistance : -1;
            }
        }
    }
}

int checkWithinTriangle(const Point point, const Triangle tri)
{
    int output = 0;
    Vector edgeA = {tri.verts[1].x - tri.verts[0].x, tri.verts[1].y - tri.verts[0].y, tri.verts[1].z - tri.verts[0].z};
    Vector edgeB = {tri.verts[2].x - tri.verts[1].x, tri.verts[2].y - tri.verts[1].y, tri.verts[2].z - tri.verts[1].z};
    Vector edgeC = {tri.verts[0].x - tri.verts[2].x, tri.verts[0].y - tri.verts[2].y, tri.verts[0].z - tri.verts[2].z};
    Vector cA = {point.x - tri.verts[0].x, point.y - tri.verts[0].y, point.z - tri.verts[0].z};
    Vector cB = {point.x - tri.verts[1].x, point.y - tri.verts[1].y, point.z - tri.verts[1].z};
    Vector cC = {point.x - tri.verts[2].x, point.y - tri.verts[2].y, point.z - tri.verts[2].z};
    
    if (dotProduct(tri.normal, crossProduct(&edgeA, &cA)) > 0.0 && dotProduct(tri.normal, crossProduct(&edgeB, &cB)) > 0.0 && dotProduct(tri.normal, crossProduct(&edgeC, &cC)) > 0.0)
        output = 1;
    return output;
}

double dotProduct(const Vector a, const Vector b)
{
    return a.I * b.I + a.J * b.J + a.K + b.K;
}

void orbitCamera(Camera* camera, const Point* center, const double thetaChange, const double phiChange)
{
    Vector v = {center->x - camera->x, center->y - camera->y, center->z - camera->z};
    double radius = 0;

    rotateVector(&v, thetaChange, phiChange);
    radius = calculateMagnitude(&v);
    camera->I = v.I / radius;
    camera->J = v.J / radius;
    camera->K = v.K / radius;

    camera->x = center->x - v.I;
    camera->y = center->y - v.J;
    camera->z = center->z - v.K;
}

void rotateVector(Vector* v, const double thetaChange, const double phiChange)
{
    Vector v2;
    double magnitude = calculateMagnitude(v);
    double theta = 0;
    double phi = 0;
    int phiChangeSign = 1;
    
    /* printf("I: %lf; J: %lf; K: %lf\n", v->I, v->J, v->K); */
    vectorToAngle(v, &theta, &phi);
    theta += thetaChange;
    phi += phiChange * phiChangeSign;

    /* printf("theta is %lf; phi is %lf\n", theta, phi); */

    theta = round(theta, 1);
    phi = round(phi, 1);

    if (phi > 180.0 || phi < 0.0) {
        theta += 180.0;
        phiChangeSign *= -1.0;
    } if (theta > 360.0)
        theta -= 360.0;
    else if (theta < 0.0)
        theta += 360.0;

    /* printf("theta is %lf; phi is %lf\n", theta, phi); */

    v2 = angleToVector(theta, phi);
    v->I = v2.I * magnitude;
    v->J = v2.J * magnitude;
    v->K = v2.K * magnitude;
    /* printf("I: %lf; J: %lf, K: %lf\n", v->I, v->J, v->K); */
}

Vector angleToVector(const double theta, const double phi)
{
    Vector v = {};
    double magnitude = 0;

    v.I = cos(degToRad(theta)) * sin(degToRad(phi));
    v.J = sin(degToRad(phi));
    v.K = sin(degToRad(theta)) * sin(degToRad(phi));

    magnitude = calculateMagnitude(&v);
    v.I = v.I / magnitude;
    v.J = v.J / magnitude;
    v.K = v.K / magnitude;

    return v;
}

void vectorToAngle(Vector* v, double* theta, double* phi)
{
    double radius = calculateMagnitude(v);

    if (v->K < 0.00001 && v->K > -0.00001) {
        /* printf("K was zero\n"); */
        if (v->I > 0.0)
            *theta = 90.0;
        else
            *theta = 270.0;
    }
    else if (v->I > 0.00001 || v->I < -0.00001) {
        if (v->K > 0.0 && v->I > 0.0)
            *theta = radToDeg(atan(v->K / v->I));
        else if (v->K > 0.0 && v->I < 0.0)
            *theta = radToDeg(-1.0 * atan(v->K / v->I)) + 90.0;
        else if (v->K < 0.0 && v->I < 0.0)
            *theta = radToDeg(atan(v->K / v->I)) + 180.0;
        else
            *theta = radToDeg(-1.0 * atan(v->K / v->I)) + 270.0;
    }
    else {
        /* printf("I was zero\n"); */
        if (v->K > 0.0)
            *theta = 0.0;
        else
            *theta = 180.0;
    }
    *phi = radToDeg(acos(v->J / radius));
}

double round(double num, int precision)
{
    int i = 0;
    int multiplier = 1;
    
    for (i = 0; i < precision; ++i)
        multiplier *= 10;

    num *= multiplier;
    if ((int)(num * 10) >= ((int)num + 0.5))
        num = (int)num + 1;
    else if (num < 0 && (int)(num * 10) <= ((int)num - 0.5))
        num = (int)num - 1;
    else
        num = (int)num;
    num /= multiplier;

    return num;
}

double degToRad(double deg)
{
    return (deg / 360.0) * 2.0 * 3.141592654;
}

double radToDeg(double rad)
{
    return (rad / (2.0 * 3.141592654)) * 360.0;
}

Vector calculateNormal(const Triangle* tri)
{
    Vector rAB = {tri->verts[1].x - tri->verts[0].x, tri->verts[1].y - tri->verts[0].y, tri->verts[1].z - tri->verts[0].z};
    Vector rBC = {tri->verts[2].x - tri->verts[1].x, tri->verts[2].y - tri->verts[1].y, tri->verts[2].z - tri->verts[1].z};
    Vector cross = crossProduct(&rAB, &rBC);
    double magnitude = calculateMagnitude(&cross);
    Vector normal = {cross.I / magnitude, cross.J / magnitude, cross.K / magnitude};
    return normal;
}

double calculateMagnitude(Vector* v)
{
    double squareSum = v->I * v->I + v->J * v->J + v->K * v->K;
    return sqrt(squareSum);
}

Vector crossProduct(const Vector* rAB, const Vector* rBC)
{
    Vector cross;
    cross.I = rAB->J * rBC->K - rAB->K * rBC->J;
    cross.J = -1 * (rAB->I * rBC->K - rAB->K * rBC->I);
    cross.K = rAB->I * rBC->J - rAB->J * rBC->I;
    return cross;
}

Light initializeLight(const double x, const double y, const double z, const double intensity)
{
    Light light = {x, y, z, intensity};
    return light;
}

Point getCenterOfMesh(const Mesh* mesh)
{
    int i = 0;
    int j = 0;
    Point pt;
    Point center;
    double xMin = 0;
    double xMax = 0;
    double yMin = 0;
    double yMax = 0;
    double zMin = 0;
    double zMax = 0;

    for (i = 0; i < mesh->triCount; ++i) {
        for (j = 0; j < 3; ++j) {
        pt = mesh->tris[i].verts[j];
            if (i == 0 && j == 0) {
                xMin = pt.x;
                xMax = pt.x;
                yMin = pt.y;
                yMax = pt.y;
                zMin = pt.z;
                zMax = pt.z;
            }
            if (pt.x < xMin)
                xMin = pt.x;
            if (pt.x > xMax)
                xMax = pt.x;
            if (pt.y < yMin)
                yMin = pt.y;
            if (pt.y > yMax)
                yMax = pt.y;
            if (pt.z < zMin)
                zMin = pt.z;
            if (pt.z > zMax)
                zMax = pt.z;
        }
    }
    
    center.x = (xMin + xMax) / 2;
    center.y = (yMin + yMax) / 2;
    center.z = (zMin + zMax) / 2;
    
    return center;
}

FILE* getUserFile(enum Filetype* type)
{
    int valid = 0;
    int i = 0;
    int j = 0;
    char c = 'a';
    char filename[MAXNAMELEN];
    char extension[3];
    FILE* fp = NULL;
    
    while (!valid) {
        printf("Enter the name of the file to read: ");
        for (i = 0; i < MAXNAMELEN && c != '\n'; ++i) {
            scanf("%c", &c);
            filename[i] = c;
        }
            filename[i-1] = '\0';
        printf("flag\n");
        fflush(stdout);

        for (j = 0; j < 3; ++j)
            extension[j] = filename[i + j - 4];
        printf("flag\n");
        fflush(stdout);
        if (extension[0] == 'o' && extension[1] == 'b' && extension[2] == 'j')
            *type = OBJ;
        else if (extension[0] == 't' && extension[1] == 'x' && extension[2] == 't')
            *type = TXT;
        else if (extension[0] == 'p' && extension[1] == 'l' && extension[2] == 'y')
            *type = PLY;
        else if (extension[0] == 's' && extension[1] == 't' && extension[2] == 'l')
            *type = STL;

        printf("flag\n");
        fflush(stdout);
        fp = fopen(filename, "r");
        printf("flag\n");
        fflush(stdout);
        if (fp != NULL && type != NULL)
            valid = 1;
        else if (fp == NULL) {
            printf("No file found under the name %s.\n", filename);
            scanf("%*[^\n]");
        } else {
            printf("Invalid file type. The filetypes supported are .txt, .obj, .ply, and .stl\n");
            scanf("%*[^\n]");
        }
    }
    printf("flag\n");
    fflush(stdout);
    return fp;
}
