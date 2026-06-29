#include <iostream>
using namespace std;

class Box {
  public:
    Box() { ; }
    ~Box() { ; }
    Box *get_address() // 得到this的地址
    {
	return this;
    }
};

int main() {

    Box box1;
    Box box2;
    // Box* 定义指针p接受对象box的get_address()成员函数的返回值，并打印

    Box *p = box1.get_address();
    cout << p << endl;

    p = box2.get_address();
    cout << p << endl;

    return 0;
}