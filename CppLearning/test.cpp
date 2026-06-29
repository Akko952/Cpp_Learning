#include <iostream>

using namespace std;

void func(int &a, int &b) {
    int tmp = b;
    b = a;
    a = tmp;
}

int main() {
    int x, y;
    x = 10, y = 5;
    cout << &x;
    cout << "\n";
    cout << &y;
    cout << "\n";
    func(x, y);
    cout << &x;
    cout << "\n";
    cout << &y;
    return 0;
}