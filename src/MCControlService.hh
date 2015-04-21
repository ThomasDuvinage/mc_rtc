// This file is generated by omniidl (C++ backend)- omniORB_4_1. Do not edit.
#ifndef __MCControlService_hh__
#define __MCControlService_hh__

#ifndef __CORBA_H_EXTERNAL_GUARD__
#include <omniORB4/CORBA.h>
#endif

#ifndef  USE_stub_in_nt_dll
# define USE_stub_in_nt_dll_NOT_DEFINED_MCControlService
#endif
#ifndef  USE_core_stub_in_nt_dll
# define USE_core_stub_in_nt_dll_NOT_DEFINED_MCControlService
#endif
#ifndef  USE_dyn_stub_in_nt_dll
# define USE_dyn_stub_in_nt_dll_NOT_DEFINED_MCControlService
#endif






#ifdef USE_stub_in_nt_dll
# ifndef USE_core_stub_in_nt_dll
#  define USE_core_stub_in_nt_dll
# endif
# ifndef USE_dyn_stub_in_nt_dll
#  define USE_dyn_stub_in_nt_dll
# endif
#endif

#ifdef _core_attr
# error "A local CPP macro _core_attr has already been defined."
#else
# ifdef  USE_core_stub_in_nt_dll
#  define _core_attr _OMNIORB_NTDLL_IMPORT
# else
#  define _core_attr
# endif
#endif

#ifdef _dyn_attr
# error "A local CPP macro _dyn_attr has already been defined."
#else
# ifdef  USE_dyn_stub_in_nt_dll
#  define _dyn_attr _OMNIORB_NTDLL_IMPORT
# else
#  define _dyn_attr
# endif
#endif





_CORBA_MODULE OpenHRP

_CORBA_MODULE_BEG

#ifndef __OpenHRP_mMCControlService__
#define __OpenHRP_mMCControlService__

  class MCControlService;
  class _objref_MCControlService;
  class _impl_MCControlService;
  
  typedef _objref_MCControlService* MCControlService_ptr;
  typedef MCControlService_ptr MCControlServiceRef;

  class MCControlService_Helper {
  public:
    typedef MCControlService_ptr _ptr_type;

    static _ptr_type _nil();
    static _CORBA_Boolean is_nil(_ptr_type);
    static void release(_ptr_type);
    static void duplicate(_ptr_type);
    static void marshalObjRef(_ptr_type, cdrStream&);
    static _ptr_type unmarshalObjRef(cdrStream&);
  };

  typedef _CORBA_ObjRef_Var<_objref_MCControlService, MCControlService_Helper> MCControlService_var;
  typedef _CORBA_ObjRef_OUT_arg<_objref_MCControlService,MCControlService_Helper > MCControlService_out;

#endif

  // interface MCControlService
  class MCControlService {
  public:
    // Declarations for this interface type.
    typedef MCControlService_ptr _ptr_type;
    typedef MCControlService_var _var_type;

    static _ptr_type _duplicate(_ptr_type);
    static _ptr_type _narrow(::CORBA::Object_ptr);
    static _ptr_type _unchecked_narrow(::CORBA::Object_ptr);
    
    static _ptr_type _nil();

    static inline void _marshalObjRef(_ptr_type, cdrStream&);

    static inline _ptr_type _unmarshalObjRef(cdrStream& s) {
      omniObjRef* o = omniObjRef::_unMarshal(_PD_repoId,s);
      if (o)
        return (_ptr_type) o->_ptrToObjRef(_PD_repoId);
      else
        return _nil();
    }

    static _core_attr const char* _PD_repoId;

    // Other IDL defined within this scope.
    
  };

  class _objref_MCControlService :
    public virtual ::CORBA::Object,
    public virtual omniObjRef
  {
  public:
    ::CORBA::Boolean EnablePostureController();
    ::CORBA::Boolean EnableBody6dController();
    ::CORBA::Boolean change_joint(const char* jname);
    ::CORBA::Boolean joint_up();
    ::CORBA::Boolean joint_down();
    ::CORBA::Boolean set_joint_pos(const char* jname, ::CORBA::Double v);
    ::CORBA::Boolean change_ef(const char* body);
    ::CORBA::Boolean translate_ef(::CORBA::Double x, ::CORBA::Double y, ::CORBA::Double z);
    ::CORBA::Boolean rotate_ef(::CORBA::Double r, ::CORBA::Double p, ::CORBA::Double y);

    inline _objref_MCControlService()  { _PR_setobj(0); }  // nil
    _objref_MCControlService(omniIOR*, omniIdentity*);

  protected:
    virtual ~_objref_MCControlService();

    
  private:
    virtual void* _ptrToObjRef(const char*);

    _objref_MCControlService(const _objref_MCControlService&);
    _objref_MCControlService& operator = (const _objref_MCControlService&);
    // not implemented

    friend class MCControlService;
  };

  class _pof_MCControlService : public _OMNI_NS(proxyObjectFactory) {
  public:
    inline _pof_MCControlService() : _OMNI_NS(proxyObjectFactory)(MCControlService::_PD_repoId) {}
    virtual ~_pof_MCControlService();

    virtual omniObjRef* newObjRef(omniIOR*,omniIdentity*);
    virtual _CORBA_Boolean is_a(const char*) const;
  };

  class _impl_MCControlService :
    public virtual omniServant
  {
  public:
    virtual ~_impl_MCControlService();

    virtual ::CORBA::Boolean EnablePostureController() = 0;
    virtual ::CORBA::Boolean EnableBody6dController() = 0;
    virtual ::CORBA::Boolean change_joint(const char* jname) = 0;
    virtual ::CORBA::Boolean joint_up() = 0;
    virtual ::CORBA::Boolean joint_down() = 0;
    virtual ::CORBA::Boolean set_joint_pos(const char* jname, ::CORBA::Double v) = 0;
    virtual ::CORBA::Boolean change_ef(const char* body) = 0;
    virtual ::CORBA::Boolean translate_ef(::CORBA::Double x, ::CORBA::Double y, ::CORBA::Double z) = 0;
    virtual ::CORBA::Boolean rotate_ef(::CORBA::Double r, ::CORBA::Double p, ::CORBA::Double y) = 0;
    
  public:  // Really protected, workaround for xlC
    virtual _CORBA_Boolean _dispatch(omniCallHandle&);

  private:
    virtual void* _ptrToInterface(const char*);
    virtual const char* _mostDerivedRepoId();
    
  };


_CORBA_MODULE_END



_CORBA_MODULE POA_OpenHRP
_CORBA_MODULE_BEG

  class MCControlService :
    public virtual OpenHRP::_impl_MCControlService,
    public virtual ::PortableServer::ServantBase
  {
  public:
    virtual ~MCControlService();

    inline ::OpenHRP::MCControlService_ptr _this() {
      return (::OpenHRP::MCControlService_ptr) _do_this(::OpenHRP::MCControlService::_PD_repoId);
    }
  };

_CORBA_MODULE_END



_CORBA_MODULE OBV_OpenHRP
_CORBA_MODULE_BEG

_CORBA_MODULE_END





#undef _core_attr
#undef _dyn_attr



inline void
OpenHRP::MCControlService::_marshalObjRef(::OpenHRP::MCControlService_ptr obj, cdrStream& s) {
  omniObjRef::_marshal(obj->_PR_getobj(),s);
}




#ifdef   USE_stub_in_nt_dll_NOT_DEFINED_MCControlService
# undef  USE_stub_in_nt_dll
# undef  USE_stub_in_nt_dll_NOT_DEFINED_MCControlService
#endif
#ifdef   USE_core_stub_in_nt_dll_NOT_DEFINED_MCControlService
# undef  USE_core_stub_in_nt_dll
# undef  USE_core_stub_in_nt_dll_NOT_DEFINED_MCControlService
#endif
#ifdef   USE_dyn_stub_in_nt_dll_NOT_DEFINED_MCControlService
# undef  USE_dyn_stub_in_nt_dll
# undef  USE_dyn_stub_in_nt_dll_NOT_DEFINED_MCControlService
#endif

#endif  // __MCControlService_hh__

