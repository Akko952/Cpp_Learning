#include <iostream>

using namespace std;

class Line {
  public:
    Line(); // 构造函数声明
    void setLength(double len);
    double getLength() const;

  private:
    double length;
};

// 构造函数定义
Line::Line() {
    cout << "Object is being created" << endl;
    length = 0.0; // 显式初始化成员变量
}

void Line::setLength(double len) { length = len; }

double Line::getLength() const { return length; }

// 程序的主函数
int main() {
    Line line;

    // 设置长度
    line.setLength(6.0);
    cout << "Length of line : " << line.getLength() << endl;

    return 0;
}